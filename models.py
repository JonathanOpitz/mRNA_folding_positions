import torch.nn as nn
import torch
import torch.nn.functional as F
    

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer= None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class mfe_conv_sim(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_latent):
        super(mfe_conv_sim, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        leakyrelu = nn.LeakyReLU(0.2, True)
        downsample = nn.MaxPool1d(3, 2, 1)
        self.drop = nn.Dropout1d()

        self.encoder = nn.Sequential(
            nn.Conv1d(dim_in, dim_hidden//2, 3, 1, 1),
            nn.BatchNorm1d(dim_hidden//2),
            leakyrelu,
            nn.Conv1d(dim_hidden//2, dim_hidden//2, 3, 1, 1),
            nn.BatchNorm1d(dim_hidden//2),
            leakyrelu,
            downsample,
            BasicBlock(dim_hidden//2, dim_hidden//2, stride=2, downsample=downsample),
            nn.Conv1d(dim_hidden//2, dim_hidden//2, 1, 1, 0),

            BasicBlock(dim_hidden//2, dim_hidden//2, stride=2, downsample=downsample),
            nn.Conv1d(dim_hidden//2, dim_hidden//2, 1, 1, 0),

            BasicBlock(dim_hidden//2, dim_hidden//2, stride=2, downsample=downsample),
            nn.Conv1d(dim_hidden//2, dim_hidden, 1, 1, 0),

            BasicBlock(dim_hidden, dim_hidden, stride=2, downsample=downsample),
            nn.Conv1d(dim_hidden, dim_hidden, 1, 1, 0),

            BasicBlock(dim_hidden, dim_hidden, stride=2, downsample=downsample),
            nn.Conv1d(dim_hidden, dim_hidden, 1, 1, 0),

            BasicBlock(dim_hidden, dim_hidden, stride=2, downsample=downsample),
            nn.Conv1d(dim_hidden, dim_hidden, 1, 1, 0),

            BasicBlock(dim_hidden, dim_hidden, stride=1),
            nn.Conv1d(dim_hidden, dim_hidden, 1, 1, 0),
            nn.Dropout1d(p=0.5),
            BasicBlock(dim_hidden, dim_hidden, stride=1),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )

        self.out = nn.Sequential(
            nn.Linear(dim_hidden, 512),
            leakyrelu,
            nn.Linear(512, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.out(x)
        return out


if __name__ == '__main__':
    pass