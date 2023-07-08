import timm
import torch
from cirtorch.layers.pooling import GeM
from torch import nn


class ResNetOfftheShelfGeM(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=False) -> None:
        super(ResNetOfftheShelfGeM, self).__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )
        self.pooling = GeM()

    def forward(self, x):
        b = x.size(0)
        x = self.backbone(x)[-1]
        x = self.pooling(x).view(b, -1)
        return x


if __name__ == "__main__":
    model = ResNetOfftheShelfGeM(pretrained=True)
    img = torch.rand(8, 3, 128, 128)
    out = model(img)
    print(out.size())
