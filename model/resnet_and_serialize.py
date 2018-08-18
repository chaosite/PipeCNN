import math
import torch.nn as nn
import torch
import os
from collections import OrderedDict
import struct

__all__ = ['resnet_original']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_()  # The original initialization in class _BatchNorm
            m.bias.data.zero_()  # The original initialization in class _BatchNorm

        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features
            m.weight.data.normal_(0, math.sqrt(2. / n))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 act_bitwidth=32):
        super(BasicBlock, self).__init__()
        if isinstance(act_bitwidth, list):
            assert (len(act_bitwidth) == 2)
            self.act_bitwidth = act_bitwidth
        else:
            self.act_bitwidth = [act_bitwidth] * 2
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_bitwidth=32, act_quant=False, act_noise=False,
                 uniq=True):
        super(Bottleneck, self).__init__()
        if isinstance(act_bitwidth, list):
            assert (len(act_bitwidth) == 3)
            self.act_bitwidth = act_bitwidth
        else:
            self.act_bitwidth = [act_bitwidth] * 3

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(self, quant=False, noise=False, bitwidth=32, step=2, quant_edges=True, act_noise=True,
                 step_setup=[15, 9], act_bitwidth=32, act_quant=False, uniq=True):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):
    def __init__(self, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        block_size = 2 if block is BasicBlock else 3
        depth = block_size * sum(layers) + 2
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 5e-2,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 5e-3},
            {'epoch': 60, 'lr': 5e-4, 'weight_decay': 0},
            {'epoch': 90, 'lr': 5e-5}
        ]


class ResNet_cifar10(ResNet):
    def __init__(self, num_classes=10, block=BasicBlock, depth=18, layers=[2, 2, 2, 2]):
        super(ResNet_cifar10, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # self.layer1 = self._make_layer(block, 16, n, act_bitwidth=self.act_bitwidth[1:2 * n + 1], act_quant=act_quant)
        # self.layer2 = self._make_layer(block, 32, n, stride=2, act_bitwidth=self.act_bitwidth[2 * n + 1:4 * n + 1],
        #                                act_quant=act_quant)
        # self.layer3 = self._make_layer(block, 64, n, stride=2, act_bitwidth=self.act_bitwidth[4 * n + 1:6 * n + 1],
        #                                act_quant=act_quant)
        # self.layer4 = lambda x: x
        # self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64, num_classes)

        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]


def resnet_original(**kwargs):
    num_classes, depth, dataset = map(kwargs.get, ['num_classes', 'depth', 'dataset'])
    dataset = dataset or 'imagenet'

    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 56
        return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth, layers=[2, 2, 2, 2])


def load_pretrained():
    pre_trained = "./model_best.pth.tar"
    target_model_config = {'input_size': 32, 'dataset': 'cifar10', 'depth': 18}
    model = resnet_original(**target_model_config)
    model = torch.nn.DataParallel(model, [0])  # Load to GPU 0


    print("=> loading checkpoint '{}'".format(pre_trained))
    checkpoint = torch.load(pre_trained, map_location='cpu')
    start_epoch = checkpoint['epoch'] - 1
    best_test = checkpoint['best_prec1']
    checkpoint_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def serialize_conv_to_file(conv, output_file):
     #### write first conv to file ####
    weights_tensor = conv.weight
    weights = weights_tensor.detach().numpy()

    ## weights ##
    for idx_0 in range(weights_tensor.shape[0]):
        for idx_1 in range(weights_tensor.shape[1]):
            for idx_2 in range(weights_tensor.shape[2]):
                for idx_3 in range(weights_tensor.shape[3]):
                    raw_float = struct.pack("f", weights[idx_0][idx_1][idx_2][idx_3])
                    output_file.write(raw_float)

    ## bias - padding of zeros ##
    raw_zero = struct.pack("f", 0)
    for idx_0 in range(weights_tensor.shape[0]):
        for idx_1 in range(weights_tensor.shape[1]):
            for idx_2 in range(weights_tensor.shape[2]):
                for idx_3 in range(weights_tensor.shape[3]):
                    output_file.write(raw_zero)

def serialize_bn_to_file(bn, output_file):
    weights_tensor = bn.weight
    weights = weights_tensor.detach().numpy()
    running_mean_tensor = bn.running_mean
    running_mean = running_mean_tensor.detach().numpy()
    bias_tensor = bn.bias
    bias = bias_tensor.detach().numpy()
    epsilon = bn.eps

    ## mult ##
    for idx_0 in range(running_mean_tensor.shape[0]):
        sqrt_val = math.sqrt(running_mean[idx_0] + epsilon)
        mult = weights[idx_0] / sqrt_val
        raw_mult = struct.pack("f", mult)
        output_file.write(raw_mult)

    ## add ##
    for idx_0 in range(running_mean_tensor.shape[0]):
        sqrt_val = math.sqrt(running_mean[idx_0] + epsilon)
        add = (sqrt_val * bias[idx_0] - running_mean[idx_0]) / sqrt_val
        raw_add = struct.pack("f", add)
        output_file.write(raw_add)

def serialize_basic_block_to_file(basic_block, output_file):
    serialize_conv_to_file(basic_block._modules['conv1'], output_file)
    serialize_bn_to_file(basic_block._modules['bn1'], output_file)
    serialize_conv_to_file(basic_block._modules['conv2'], output_file)
    serialize_bn_to_file(basic_block._modules['bn2'], output_file)

    if 'downsample' in basic_block._modules.keys():
        serialize_conv_to_file(basic_block._modules['downsample'][0],
                               output_file)
        serialize_bn_to_file(basic_block._modules['downsample'][1],
                             output_file)

def serialize_layer_to_file(layer, output_file):
    serialize_basic_block_to_file(layer[0], output_file)
    serialize_basic_block_to_file(layer[1], output_file)

def serialize_weights(model, output_file_name):
    f = open(output_file_name, "wb")

    serialize_conv_to_file(model._modules['module']._modules['conv1'], f)
    serialize_bn_to_file(model._modules['module']._modules['bn1'], f)

    serialize_layer_to_file(model._modules['module']._modules['layer1'], f)
    serialize_layer_to_file(model._modules['module']._modules['layer2'], f)
    serialize_layer_to_file(model._modules['module']._modules['layer3'], f)
    serialize_layer_to_file(model._modules['module']._modules['layer4'], f)

    f.close()

def verbose(conv_layer):
    print("in_channels=" + str(conv_layer.in_channels))
    print("out_channels=" + str(conv_layer.out_channels))
    # print("kernel_size=" str(conv_layer.kernel_size[0]) + "," + str(conv_layer.kernel_size[1]))
    print("padding=" + str(conv_layer.padding))
    print("stride=" + str(conv_layer.stride))

if __name__ == "__main__":
    load_pretrained()
    # load_model(model, checkpoint)
    print(model)
