from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from modules.LRN import LocalResponseNorm
import torch.utils.model_zoo as model_zoo

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

class AlexNetModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        # pool_size, pool_stride
        self.conv_pool_params = {
            (3, 96, 11, 4, 2, 3, 2),
            (96, 256, 5, 1, 2, 3, 2),
        }
        self.conv_params = {
            (256, 384, 11, 1, 1),
            (384, 384, 11, 1, 1),
            (384, 256, 11, 1, 1),
        }
        self.fc_params = {
            (256 * 3 * 3, 4096),
            (4096, 4096),
            (4096, num_classes, 'True')
        }
        self.features = nn.Sequential(
            *nn.ModuleList(
                self.conv_pool_layer(params)
                for params in self.conv_pool_params
            ),
            *nn.ModuleList(
                self.conv_layer(params) 
                for params in self.conv_params
            )
        )
        self.classifier = nn.Sequential(
            *nn.ModuleList(
                self.fc_layer(params)
                for params in self.fc_params
            )
        )

    def conv_pool_layer(in_channels, out_channels, kernel_size, stride,
                        padding, pool_size, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            LocalResponseNorm(n=5, alpha=1e-4, beta=0.75, k=2)
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        )

    def conv_layer(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )

    def fc_layer(in_features, out_features, last=False):
        if last:
            return nn.Linear(in_features, out_features)
        else:
            return nn.Sequential(
                nn.Dropout()
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        # pool_size, pool_stride
        self.conv_pool_params = {
            (3, 96, 11, 4, 2, 3, 2),
            (96, 256, 5, 1, 2, 3, 2),
        }
        self.conv_params = {
            (256, 384, 11, 1, 1),
            (384, 384, 11, 1, 1),
            (384, 256, 11, 1, 1),
        }
        self.fc_params = {
            (256 * 3 * 3, 4096),
            (4096, 4096),
            (4096, num_classes, 'True')
        }
        self.features = nn.Sequential(
            *nn.ModuleList(
                self.conv_pool_layer(params)
                for params in self.conv_pool_params
            ),
            *nn.ModuleList(
                self.conv_layer(params) 
                for params in self.conv_params
            )
        )
        self.classifier = nn.Sequential(
            *nn.ModuleList(
                self.fc_layer(params)
                for params in self.fc_params
            )
        )

    def conv_pool_layer(in_channels, out_channels, kernel_size, stride,
                        padding, pool_size, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            LocalResponseNorm(n=5, alpha=1e-4, beta=0.75, k=2)
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        )

    def conv_layer(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )

    def fc_layer(in_features, out_features, last=False):
        if last:
            return nn.Linear(in_features, out_features)
        else:
            return nn.Sequential(
                nn.Dropout()
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, **kwargs):
    """
    :param pretrained: use weights from model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model








