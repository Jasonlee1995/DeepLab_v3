import torch, copy
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F

from PIL import Image

        
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, atrous_rates):
        super(ASPP, self).__init__()
        convs = [nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())]
        for rate in atrous_rates:
            convs.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False), 
                                       nn.BatchNorm2d(out_ch), nn.ReLU()))
        self.convs = nn.ModuleList(convs)
        
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())
        
        self.concat = nn.Sequential(nn.Conv2d((len(self.convs)+1)*out_ch, out_ch, kernel_size=1, stride=1, bias=False), 
                                    nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x):
        n, c, h, w = x.shape
        xs = [self.convs[i](x) for i in range(len(self.convs))]
        xs.append(F.resize(self.pool(x), (h, w), Image.BILINEAR))
        output = torch.cat(xs, dim=1)
        output = self.concat(output)
        return output
    
    
class ResNet_ASPP(nn.Module):
    def __init__(self, num_classes, multi_grid, atrous_rates, output_stride, scales):
        super(ResNet_ASPP, self).__init__()
        assert output_stride in [8, 16]
        
        rate, index = [2, 4, 8, 16], 0
        self.scales = scales
        
        resnet = models.resnet101(pretrained=True)
        
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
    
        for i, module in enumerate(resnet.layer4):
            num = rate[index] * multi_grid[i]
            module.conv2.padding = (num, num)
            module.conv2.dilation = (num, num)
        index += 1
        
        block5 = copy.deepcopy(resnet.layer4)
        block6 = copy.deepcopy(resnet.layer4)
        block7 = copy.deepcopy(resnet.layer4)
        blocks = [block5, block6, block7]
            
        for block in blocks:
            for i, module in enumerate(block):
                if i == 0:
                    module.conv1 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    module.downsample[0] = nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                num = rate[index] * multi_grid[i]
                module.conv2.padding = (num, num)
                module.conv2.dilation = (num, num)
            index += 1
            
        fc = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False)
            
        self.model = torch.nn.Sequential(*list(resnet.children())[:-2], *blocks,
                                         ASPP(2048, 256, atrous_rates),
                                         fc)

    def forward(self, x):
        n, c, h, w = x.shape
        if self.training:
            output = self.model(x)
            output = F.resize(output, (h, w), Image.BILINEAR)
        else:
            xs = [F.resize(x, (int(scale*h), int(scale*w)), Image.BILINEAR) for scale in self.scales]
            outputs = [self.model(x_) for x_ in xs]
            outputs = [F.resize(output, (h, w), Image.BILINEAR) for output in outputs]
            outputs = torch.stack(outputs)
            output = torch.max(outputs, dim=0)[0]
        return output