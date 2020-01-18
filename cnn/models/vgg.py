import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.vgg import VGG as _VGG
from torchvision.models.vgg import cfgs, make_layers, load_state_dict_from_url, model_urls


class VGG19Finetune(_VGG):
    def __init__(self, num_classes, weights_path=None, progress=False, frozen_layers=None):
        if weights_path is None:
            super().__init__(make_layers(cfgs['E'], batch_norm=False), num_classes=1000, init_weights=False)
            state_dict = load_state_dict_from_url(model_urls['vgg19'], progress=progress)
            self.load_state_dict(state_dict)
            self.classifier[6] = nn.Linear(4096, num_classes)
            if frozen_layers is not None:
                if frozen_layers < 1 or frozen_layers > 16:
                    raise ValueError('frozen_layers must be between 1 and 16')
                self._freeze_layers(frozen_layers)
        else:
            print('Using given weights instead of ImageNet weights')
            super().__init__(make_layers(cfgs['E'], batch_norm=False), num_classes=num_classes, init_weights=False)
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)

    def _freeze_layers(self, frozen_layers):
        layers = 0
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):  # layer count is for conv layers only
                layers += 1
            m.requires_grad_(False)
            if layers == frozen_layers:
                break

    def inference(self, x):
        x = self.forward(x)
        x = F.softmax(x, 1)
        return x
