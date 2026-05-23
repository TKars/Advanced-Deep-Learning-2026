"""SSD с backbone ResNet18; вход 3×720×1280 (см. задание недели 2)."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

from prior_boxes import prior_boxes

# по ГОСТу,  номерной знак 520×112 мм ->  отношение сторон w/h (для горизонтального якоря).
PLATE_WH_RATIO = 520 / 112


def ssd_resnet18_cfg():
    """Карты признаков после layer2…layer4 и трёх extra-блоков (без финального 1×1).

    Якоря: меньшие min_sizes на fine-картах; среди aspect ratio — явное w/h номера (520:112).
    Число якорей на уровень как раньше (3+3+3 / 2+2 / 1+1+… по формуле prior_boxes).
    """
    ar_plate = PLATE_WH_RATIO
    return {
        "num_classes": 3,
        "feature_maps": [(90, 160), (45, 80), (23, 40), (12, 20), (6, 10), (3, 5)],
        "min_dim": 300,
        "min_sizes": [0.05, 0.10, 0.20, 0.36, 0.52, 0.68],
        "max_sizes": [0.10, 0.20, 0.36, 0.52, 0.68, 0.82],
        # 2, 3 — машины / общие прямоугольники; ar_plate — типичный горизонтальный номер
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
    cfg = cfg or ssd_resnet18_cfg() 
    p = prior_boxes(cfg)
    if device is not None:
        p = p.to(device)
    return p


class SSDResNet18(nn.Module):
    

    def __init__(self, num_classes=3, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        w = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=w)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.extras = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1),
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

        cfg = ssd_resnet18_cfg()
        self.aspect_ratios = cfg["aspect_ratios"]
        sources_channels = [128, 256, 512, 512, 256, 128]

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for ar, c in zip(self.aspect_ratios, sources_channels):
            n = _num_priors_per_cell(ar)
            self.loc_layers.append(nn.Conv2d(c, n * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(c, n * num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        sources = [x]
        x = self.layer3(x)
        sources.append(x)
        x = self.layer4(x)
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
    cfg = ssd_resnet18_cfg()
    m = SSDResNet18(num_classes=cfg["num_classes"], pretrained_backbone=False)
    x = torch.zeros(1, 3, 720, 1280)
    loc, conf = m(x)
    pri = prior_boxes(cfg)
    assert loc.shape[1] == pri.shape[0] == conf.shape[1], (loc.shape, conf.shape, pri.shape)
    return loc.shape, conf.shape, pri.shape


if __name__ == "__main__":
    print(verify_shapes())
