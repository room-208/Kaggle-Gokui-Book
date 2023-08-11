import timm
import torch
from arcface import ArcMarginProduct
from cirtorch.layers.pooling import GeM
from torch import nn


class AngularModel(nn.Module):
    def __init__(
        self,
        n_classes=7770,
        model_name="resnet34",
        margin=0.3,
        scale=30,
        fc_dim=512,
        pretrained=None,
        loss_kwargs=None,
    ):
        super(AngularModel, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
        )
        final_in_features = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        loss_kwargs = loss_kwargs or {
            "s": scale,
            "m": margin,
            "easy_margin": False,
            "ls_eps": 0.0,
        }

        self.pooling = GeM()
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

        self.final = ArcMarginProduct(fc_dim, n_classes, **loss_kwargs)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_features(x)
        logits = self.final(feature, label)
        return logits

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        # fc
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x


def setup_model(device):
    model = AngularModel(pretrained=True)
    model = model.to(device)
    return model


if __name__ == "__main__":
    model = AngularModel(pretrained=True)
    label = torch.randint(low=0, high=7770, size=(8,))
    img = torch.rand(8, 3, 128, 128)
    out = model(img, label)
    print(out.size())
