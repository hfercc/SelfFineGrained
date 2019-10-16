import torch
import torchvision

import copy
import torch.nn as nn

from utils import split_image

from resnetv2 import ResNet50 as resnet50v2

class Model(nn.Module):

    def __init__(self, args, num_classes = 200):
        super(Model, self).__init__()
        if args.arch == 'resnet50':
            self.feature = torchvision.models.resnet50(pretrained = True)
            self.fc = nn.Linear(2048, num_classes)
        elif args.arch == 'resnet50v2':
            self.feature = resnet50v2(num_classes)
            self.fc = nn.Linear(2048, num_classes)
        else:
            raise NotImplementedError

        if args.with_rotation:
            self.rotation_fc = nn.Linear(2048, 4)
        else:
            self.rotation_fc = None

        if args.with_jigsaw:
            self.jigsaw_fc = nn.Sequential(
                nn.Linear(2048 * 16, 2048),
                nn.Linear(2048, 100))
        else:
            self.jigsaw_fc = None

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

        if args.seperate_layer4:
            self.rotation_layer4_ = copy.deepcopy(self.feature.layer4)
            self.jigsaw_layer4_ = copy.deepcopy(self.feature.layer4)
        else:
            self.rotation_layer4_ = None
            self.jigsaw_layer4_ = None



    def extract_feature(self, x):
        x = self.feature.conv1(x)
        x = self.feature.bn1(x)
        x = self.feature.relu(x)
        x = self.feature.maxpool(x)
        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        return x

    def layer4(self, x):
        x = self.feature.layer4(x)
        x = self.feature.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def rotation_layer4(self, x):
        x = self.rotation_layer4_(x)
        x = self.feature.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def jigsaw_layer4(self, x):
        x = self.jigsaw_layer4_(x)
        x = self.feature.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, rotation_x = None, jigsaw_x = None):
        x = self.extract_feature(x)
        x = self.layer4(x)
        x = self.fc(x)
        if rotation_x is not None:
            rotation_x = self.extract_feature(rotation_x)
            if self.rotation_layer4_ is None:
                rotation_x = self.layer4(rotation_x)
            else:
                rotation_x = self.rotation_layer4(rotation_x)

            rotation_x = self.rotation_fc(rotation_x)

        if jigsaw_x is not None:
            jigsaw_x = self.extract_feature(jigsaw_x)
            if self.jigsaw_layer4_ is None:
                jigsaw_x = self.layer4(jigsaw_x)
            else:
                jigsaw_x = self.jigsaw_layer4(jigsaw_x)

            jigsaw_x = jigsaw_x.reshape(x.shape[0], -1)

            jigsaw_x = self.jigsaw_fc(jigsaw_x)

        return x, rotation_x, jigsaw_x


        

