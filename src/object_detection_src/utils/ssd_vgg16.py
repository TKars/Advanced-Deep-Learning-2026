import torch
import torch.nn as nn
import torchvision.models as models

from prior_boxes import prior_boxes

PLATE_WH_RATIO = 520 / 112
def ssd_vgg_cfg():
    """Конфиг приоров и карт признаков для входа 3×360×640 (см. forward SSDVGG16)."""
    ar_plate = PLATE_WH_RATIO
    return {
        'num_classes': 3,
        'feature_maps': [(45, 80), (11, 20), (6, 10), (3, 5), (2, 3), (1, 1)],
        'min_dim': 300,
        "min_sizes": [0.05, 0.10, 0.20, 0.36, 0.52, 0.68],
        "max_sizes": [0.10, 0.20, 0.36, 0.52, 0.68, 0.82],
        # +6: пары prior w/h=6 и w/h=1/6 (номер: ~6:1 по сторонам)
        'aspect_ratios': [[2, 3, ar_plate], [2, 3, ar_plate], [2, 3, ar_plate], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
    }


def num_priors_per_cell(aspect_ratio_list):
    return 2 + 2 * len(aspect_ratio_list)


def build_priors(cfg=None, device=None):
    cfg = cfg or ssd_vgg_cfg()
    p = prior_boxes(cfg)
    if device is not None:
        p = p.to(device)
    return p


class SSDVGG16(nn.Module):
    """SSD: torchvision VGG16.features до conv4_3 + хвост VGG + extra-слои; головы loc/conf."""

    def __init__(self, num_classes=3, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        w = models.VGG16_Weights.DEFAULT if pretrained_backbone else None
        vgg = models.vgg16(weights=w)
        feats = list(vgg.features.children())
        self.features_conv4 = nn.Sequential(*feats[:23])
        self.features_rest = nn.Sequential(*feats[23:])

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
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=(2, 3)),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        cfg = ssd_vgg_cfg()
        self.aspect_ratios = cfg['aspect_ratios']
        sources_channels = [512, 512, 512, 256, 128, 128]

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for ar, c in zip(self.aspect_ratios, sources_channels):
            n = num_priors_per_cell(ar)
            self.loc_layers.append(nn.Conv2d(c, n * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(c, n * num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        sources = []
        h = self.features_conv4(x)
        sources.append(h)
        h = self.features_rest(h)
        sources.append(h)
        for ex in self.extras:
            h = ex(h)
            sources.append(h)

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
    cfg = ssd_vgg_cfg()
    m = SSDVGG16(num_classes=cfg['num_classes'], pretrained_backbone=False)
    x = torch.zeros(1, 3, 360, 720)
    loc, conf = m(x)
    pri = prior_boxes(cfg)
    print(loc.size(), conf.size())
    assert loc.shape[1] == pri.shape[0] == conf.shape[1], (loc.shape, conf.shape, pri.shape)
    return loc.shape, conf.shape, pri.shape


if __name__ == '__main__':
    print(verify_shapes())
