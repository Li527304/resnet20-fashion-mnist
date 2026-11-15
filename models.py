"""CIFAR-style ResNet20, option-A shortcuts; grayscale input adaptation."""
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ('base', 'se', 'ds', 'gelu', 'gn')


class SE(nn.Module):
    def __init__(self, c, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(1,c//reduction)), nn.ReLU(),
                                nn.Linear(max(1,c//reduction), c), nn.Sigmoid())
    def forward(self, x):
        return x*self.fc(x.mean((2,3)))[:,:,None,None]


def conv(a,b,stride=1):
    return nn.Conv2d(a,b,3,stride,1,bias=False)


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, a, b, stride=1):
        super().__init__(
            nn.Conv2d(a, a, 3, stride, 1, groups=a, bias=False),
            nn.Conv2d(a, b, 1, bias=False),
        )


def norm(channels, variant):
    if variant == 'gn':
        return nn.GroupNorm(8, channels)
    return nn.BatchNorm2d(channels)


def activation(variant):
    return nn.GELU() if variant == 'gelu' else nn.ReLU()


class Block(nn.Module):
    def __init__(self,a,b,stride,variant):
        super().__init__()
        self.stride,self.a,self.b=stride,a,b
        convolution = DepthwiseSeparableConv if variant == 'ds' else conv
        self.conv1 = convolution(a, b, stride)
        self.bn1 = norm(b, variant)
        self.conv2 = convolution(b, b, 1)
        self.bn2 = norm(b, variant)
        self.activation = activation(variant)
        self.se=SE(b) if variant=='se' else nn.Identity()
    def forward(self,x):
        y=self.activation(self.bn1(self.conv1(x)))
        y=self.se(self.bn2(self.conv2(y)))
        shortcut=x[:,:,::self.stride,::self.stride]
        if self.a!=self.b:
            pad=(self.b-self.a)//2
            shortcut=F.pad(shortcut,(0,0,0,0,pad,pad))
        return self.activation(y+shortcut)


class ResNet20(nn.Module):
    def __init__(self,variant='base'):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(variant)
        self.stem=nn.Sequential(conv(1,16),norm(16,variant),activation(variant))
        blocks=[]; a=16
        for stage,b in enumerate((16,32,64)):
            for i in range(3):
                blocks.append(Block(a,b,2 if stage>0 and i==0 else 1,variant)); a=b
        self.blocks=nn.Sequential(*blocks)
        self.fc=nn.Linear(64,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x):
        return self.fc(self.blocks(self.stem(x)).mean((2,3)))


def profile(model,device):
    """MACs count only Conv2d/Linear; one MAC = one multiply-accumulate."""
    total=[0]; handles=[]
    def hook(m,inp,out):
        if isinstance(m,nn.Conv2d):
            total[0]+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        else: total[0]+=out.numel()*m.in_features
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)): handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad(): model(torch.zeros(1,1,32,32,device=device))
    for h in handles: h.remove()
    return sum(p.numel() for p in model.parameters()),total[0]
