import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from prior_boxes import prior_boxes

# ГОСТ: номерной знак 520×112 мм → отношение сторон w/h
PLATE_WH_RATIO = 520 / 112


def ssd_convnext_tiny_cfg():
    """Карты признаков после Stage-2…Stage-4 ConvNeXt-Tiny и трёх extra-блоков.

    Обратите внимание: из-за особенностей свёрток downsampling (kernel=2) в ConvNeXt
    высота H=45 уменьшается ровно до H=22, а не 23 как в ResNet.
    """
    ar_plate = PLATE_WH_RATIO
    return {
        "num_classes": 3,
        # Карты: (90,160) -> Stage2, (45,80) -> Stage3, (22,40) -> Stage4
        # (11,20), (6,10), (3,5) -> три слоя extras
        "feature_maps": [(90, 160), (45, 80), (22, 40), (11, 20), (6, 10), (3, 5)],
        "min_dim": 300,
        "min_sizes": [0.05, 0.10, 0.20, 0.36, 0.52, 0.68],
        "max_sizes": [0.10, 0.20, 0.36, 0.52, 0.68, 0.82],
        "aspect_ratios": [
            [2, 3, ar_plate],
            [2, 3, ar_plate],
            [2, 3, ar_plate],
            [2, ar_plate],
            [2],
            [2],
        ],
        "variance": [0.1, 0.2],
        "clip": True,
    }


def _num_priors_per_cell(aspect_ratio_list):
    return 2 + 2 * len(aspect_ratio_list)


def build_priors(cfg=None, device=None):
    cfg = cfg or ssd_convnext_tiny_cfg()
    p = prior_boxes(cfg)
    if device is not None:
        p = p.to(device)
    return p


class SSDConvNeXtTiny(nn.Module):
    
    def __init__(self, num_classes=3, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        w = ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
        backbone = convnext_tiny(weights=w)
        
        # Получаем полный ModuleList backbone'а
        # features[3] -> Stage 2 (192 channels)
        # features[5] -> Stage 3 (384 channels)
        # features[7] -> Stage 4 (768 channels)
        self.features = backbone.features

        # Дополнительные блоки под выход Stage 4 ConvNeXt-Tiny (в котором 768 каналов).
        self.extras = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(768, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        cfg = ssd_convnext_tiny_cfg()
        self.aspect_ratios = cfg["aspect_ratios"]
        
        # Каналы на выходных картах: Stage2, Stage3, Stage4, Extra1, Extra2, Extra3
        sources_channels = [192, 384, 768, 512, 256, 128]

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for ar, c in zip(self.aspect_ratios, sources_channels):
            n = _num_priors_per_cell(ar)
            self.loc_layers.append(nn.Conv2d(c, n * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(c, n * num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        sources = []
        
        # Проходим по секвенированным слоям ConvNeXt
        # Индексы 3, 5, 7 соответствуют окончаниям Stage 2, 3 и 4
        for i, module in enumerate(self.features):
            x = module(x)
            if i in [3, 5, 7]:
                sources.append(x)
                
        for ex in self.extras:
            x = ex(x)
            sources.append(x)
        loc_list, conf_list = [], []
        for loc_layer, conf_layer, feat in zip(self.loc_layers, self.conf_layers, sources):
            loc_list.append(loc_layer(feat).permute(0, 2, 3, 1).contiguous())
            conf_list.append(conf_layer(feat).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_list], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf_list], dim=1)
        
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return loc, conf


def verify_shapes():
    cfg = ssd_convnext_tiny_cfg()
    m = SSDConvNeXtTiny(num_classes=cfg["num_classes"], pretrained_backbone=False)
    x = torch.zeros(1, 3, 720, 1280)
    
    loc, conf = m(x)
    pri = prior_boxes(cfg)
    
    assert loc.shape[1] == pri.shape[0] == conf.shape[1], (loc.shape, conf.shape, pri.shape)
    return loc.shape, conf.shape, pri.shape


if __name__ == "__main__":
    print(verify_shapes())