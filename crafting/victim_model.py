import torch
import json
import torch.nn as nn

from codebase.networks.nsganetv2 import NSGANetV2


class VictimModel(nn.Module):
    def __init__(self, model_config_path, model_weights_path, device, arch=None, num_classes=10):
        super(VictimModel, self).__init__()
        
        net_config = json.load(open(model_config_path))
        
        self.model = NSGANetV2.build_from_config(net_config)
        try:
            self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        except:
            self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu')['state_dict'])

        self.model = self.model.to(device)
        self._target_parameters = [p for p in self.model.parameters()]

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def target_parameters(self):
        return self._target_parameters

    def save(self, path):
        torch.save(self.model.state_dict(), path)