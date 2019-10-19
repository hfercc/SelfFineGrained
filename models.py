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

        if args.dataset == 'cifar':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)

        if args.with_rotation:
            self.rotation_fc = nn.Linear(2048, 4)
        else:
            self.rotation_fc = None

        if args.with_jigsaw:
            self.jigsaw_fc = nn.Sequential(
                nn.Linear(2048 * 16, 2048),
                nn.Linear(2048, 30))
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

def split_resnet50(model):
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3
    )


class SelfEnsembleModel(nn.Module):

    def __init__(self, args, num_of_branches, num_classes = 200):

        super(SelfEnsembleModel, self).__init__()
        self.num_of_branches = num_of_branches
        self.args = args

        if args.arch == 'resnet50v1':
            self.branches = [split_resnet50(torchvision.models.resnet50(pretrained = True)) for _ in range(num_of_branches)]
            self.layer4 = torchvision.models.resnet50(pretrained = True).layer4 
            self.fc = nn.Linear(2048, num_classes)
        elif args.arch == 'resnet50v2':
            self.branches = [split_resnet50(resnet50v2()) for _ in range(num_of_branches)]
            self.layer4 = resnet50v2().layer4 
            self.fc = nn.Linear(2048, num_classes)
        

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.gate = None # Initialize in forward()

        for i in self.branches:
            i.cuda()


    def _load(self, files):
        for i in range(self.num_of_branches):
            state_dict = self.branches[i].state_dict()
            new_state_dict = torch.load(files[i])
            state_dict.update(new_state_dict)
            self.branches[i].load_state_dict(state_dict)

    def forward(self, x):
        # x: List
        feature_maps = []
        
        if self.gate is None:
            self.gate = nn.Parameter(torch.ones(self.num_of_branches).cuda(self.args.gpu) * 1.0 / self.num_of_branches)

        #print(self.gate.grad)
        for i in range(self.num_of_branches):
            feature_map = self.branches[i](x[i]).unsqueeze(0)
            feature_maps.append(feature_map * self.gate[i])
        feature_maps = torch.sum(torch.cat(feature_maps, 0), 0)

        feature_maps = self.layer4(feature_maps)
        feature_maps = self.avgpool(feature_maps)
        feature_maps = torch.flatten(feature_maps, 1)

        feature_maps = self.fc(feature_maps)

        return feature_maps



        

        

