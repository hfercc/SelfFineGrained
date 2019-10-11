import torch
import torchvision

import torch.nn as nn

from resnetv2 import ResNet50 as resnet50v2

class Model(nn.Module):

    def __init__(self, args, num_classes = 200):

        if args.arch == 'resnet50':
            self.feature = torchvision.models.resnet50(pretrained = True)
            self.fc = nn.Linear(2048, num_classes)
        elif args.arch == 'resnet50v2':
            self.feature = ResNet50v2(num_classes)
            self.fc = nn.Linear(2048, num_classes)
        else:
            raise NotImplementedError

        if args.with_rotation:
            self.fc_rotation = nn.Linear(2048, 4)
            self.with_rotation = True

        if args.load_weights is not None:
            try:
                state_dict = model.state_dict()
                load_state_dict = torch.load(args.load_weights)['P_state']
                state_dict.update(load_state_dict)
                model.load_state_dict(state_dict)

            except RuntimeError:
                model_loaded = torch.load(args.load_weights)
                data_dict = model_loaded['P_state']
                state_dict = model.state_dict()
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in data_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                new_state_dict.pop('conv1.weight', None)
                new_state_dict.pop('conv1.bias', None)
                state_dict.update(new_state_dict)
                model.load_state_dict(state_dict)
                del new_state_dict
                del data_dict

    def extract_feature(self, x):
        x = self.feature.conv1(x)
        x = self.feature.bn1(x)
        x = self.feature.relu(x)
        x = self.feature.maxpool(x)
        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        x = self.feature.layer4(x)
        x = self.feature.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x, rotation_input = None):
        x = self.extract_feature(x)
        x = self.fc(x)
        if rotation_input is not None:
            rotation_input = self.extract_feature(rotation_input)
            rotation_input = self.fc_rotation(rotation_input)

        return x, rotation_input


        

