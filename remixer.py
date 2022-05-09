import torch
from torch.functional import split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import revtorch as rv

class ElementWiseMLP(nn.Module):
    """Some Information about ElementWiseMLP"""
    def __init__(self, dim, activation='gelu'):
        super(ElementWiseMLP, self).__init__()
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ln(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class MixerMLP(nn.Module):
    """Some Information about MixerMLP"""
    def __init__(self, dim, activation='gelu'):
        super(MixerMLP, self).__init__()
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.ln(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = x.swapaxes(1, 2)
        return x

# input: [batch_size, seq_len, dim]
# output: [batch_size, seq_len, dim]
class ReMixer(nn.Module):
    """Some Information about ReMixer"""
    def __init__(self, num_patch, dim, activation='gelu', num_layers=1):
        super(ReMixer, self).__init__()
        self.sequenece = rv.ReversibleSequence(nn.ModuleList([rv.ReversibleBlock(MixerMLP(num_patch, activation), ElementWiseMLP(dim, activation), split_along_dim=2) for _ in range(num_layers)]))
    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=2)
        x = self.sequenece(x)
        x1, x2 = torch.chunk(x, 2, dim=2)
        x = (x1 + x2) / 2
        return x


# input: [batch_size, channels, height, weight]
# output: [batch_size, seq_len, patch_dim]
class Image2Patch(nn.Module):
    """Some Information about Image2Patch"""
    def __init__(self, channels, image_size, patch_size):
        super(Image2Patch, self).__init__()
        if type(patch_size) == int:
            patch_size = [patch_size, patch_size] # [height, width]
        self.patch_size = patch_size
        if type(image_size) == int:
            image_size = [image_size, image_size] # [height, width]
        self.image_size = image_size
        self.channels = channels
        self.num_patch = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]

    def forward(self, x):
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.swapaxes(1, 2)
        return x

# input: [batch_size, seq_len, patch_dim]
# output: [batch_size, channels, Height, Width]
class Patch2Image(nn.Module):
    """Some Information about Patch2Image"""
    def __init__(self, channels, image_size, patch_size):
        super(Patch2Image, self).__init__()
        if type(patch_size) == int:
            patch_size = [patch_size, patch_size] # [height, width]
        self.patch_size = patch_size
        if type(image_size) == int:
            image_size = [image_size, image_size] # [height, width]
        self.image_size = image_size
        self.channels = channels

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = F.fold(x, output_size=self.image_size, kernel_size=self.patch_size, stride=self.patch_size)
        return x

# this module Only supports square images.
# input: [batch_size, channels, height, width]
# output: [batch_size, classes]
class ReMixerImageClassificator(nn.Module):
    """Some Information about RemixerImageClassificator"""
    def __init__(self, channels=3, image_size=256, patch_size=16, classes=10, dim=512, num_layers=12, activation='gelu'):
        super(ReMixerImageClassificator, self).__init__()
        self.image2patch = Image2Patch(channels, image_size, patch_size)
        num_patch = (image_size // patch_size) ** 2
        dim_patch = patch_size ** 2 * channels
        self.embedding = nn.Linear(dim_patch, dim)
        self.remixer = ReMixer(num_patch, dim, activation, num_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dim2class = nn.Linear(dim, classes)
    def forward(self, x):
        x = self.image2patch(x)
        x = self.embedding(x)
        x = self.remixer(x)
        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = x.squeeze(2)
        x = self.dim2class(x)
        return x

# this module Only supports square images.
# input: [batch_size, feature_dim]
# output: [batch_size, channels, height, width]
class ReMixerImageGenerator(nn.Module):
    """Some Information about ReMixerImageGenerator"""
    def __init__(self, feature_dim=1024, channels=3, image_size=256, patch_size=16, dim=512, num_layers=12, activation='gelu'):
        super(ReMixerImageGenerator, self).__init__()
        self.num_patch = (image_size // patch_size) ** 2
        self.dim_mixer = dim
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patch,dim))
        self.feature2patch = nn.Linear(feature_dim, dim)
        self.remixer = ReMixer(self.num_patch, dim, activation, num_layers)
        self.to_channels = nn.Linear(dim, channels * patch_size ** 2)
        self.patch2image = Patch2Image(channels, image_size, patch_size)
    def forward(self, x):
        x = self.feature2patch(x)
        x = torch.repeat_interleave(x, self.num_patch, dim=1)
        x = x.reshape(x.shape[0], self.num_patch, self.dim_mixer)
        x = x + self.positional_embedding
        x = self.remixer(x)
        x = self.to_channels(x)
        x = self.patch2image(x)
        return x


# this module Only supports square images.
# input: [batch_size, input_channels, height, width]
# output: [batch_size, output_channels, height, width]
class ReMixerImage2Image(nn.Module):
    """Some Information about ReMixerImage2Image"""
    def __init__(self, input_channels, output_channels, image_size, patch_size, dim=512, num_layers=12, activation='gelu'):
        super(ReMixerImage2Image, self).__init__()
        num_patch = (image_size // patch_size) ** 2
        self.image2patch = Image2Patch(input_channels, image_size, patch_size)
        self.embedding = nn.Linear(input_channels * patch_size ** 2, dim)
        self.remixer = ReMixer(num_patch, dim, activation, num_layers)
        self.unembedding = nn.Linear(dim, output_channels * patch_size ** 2)
        self.patch2image = Patch2Image(output_channels, image_size, patch_size)
    def forward(self, x):
        x = self.image2patch(x)
        x = self.embedding(x)
        x = self.remixer(x)
        x = self.unembedding(x)
        x = self.patch2image(x)
        return x

class SpatialShift2d(nn.Module):
    def __init__(self, channels, padding_mode='replicate'):
        super(SpatialShift2d, self).__init__()
        qc = channels // 4
        self.num_shift_left = qc
        self.num_shift_right = qc
        self.num_shift_up = qc
        self.num_shift_down = channels - qc*3
        self.padding_mode = padding_mode

    def forward(self,x):
        # input: [batch_size, channels, height, width]
        _l, _r, _u, _d = self.num_shift_left, self.num_shift_right, self.num_shift_up, self.num_shift_down
        x = F.pad(x, (1,1,1,1), self.padding_mode) # pad
        l, r, u, d = torch.split(x, [_l, _r, _u, _d], dim=1) # split
        # shift
        l = l[:, :, 1:-1, 0:-2]
        r = r[:, :, 1:-1, 2:  ]
        u = u[:, :, 0:-2, 1:-1]
        d = d[:, :, 2:  , 1:-1]
        # concatenate channelwise shifted tensors
        x = torch.cat([l,r,u,d], dim=1)
        return x

# Reversible S2-MLP layer stack.
# input: [batch_size, channels, height, width]
# output: [batch_size, channels, height, width]
class ReS2MLP2d(nn.Module):
    def __init__(self, channels, image_size=[28, 28], activation='gelu', norm='layernorm', num_layers=1, padding_mode='replicate'):
        super(ReS2MLP2d, self).__init__()
        if type(image_size) != int:
            h, w = image_size[0], image_size[1]
        else:
            h, w = image_size, image_size
        # activation module
        if activation == 'gelu':
            act = nn.GELU
        elif activation == 'relu':
            act = nn.ReLU
        elif activation == 'leakyrelu':
            act == nn.LeakyReLU

        if norm == 'layernorm':
            def norm_init(c, h, w):
                return nn.LayerNorm([c, h, w])
        
        # initialize layer stack
        self.seq = rv.ReversibleSequence(
                nn.ModuleList([
                    rv.ReversibleBlock(
                        nn.Sequential( # f-block
                            nn.Conv2d(channels, channels, 1, 1, 0), # Fully-Connected 1
                            act(),
                            SpatialShift2d(channels, padding_mode),
                            nn.Conv2d(channels, channels, 1, 1, 0), # Fully-Connected 2
                            norm_init(channels,h,w),
                            ),
                        nn.Sequential( # g_block
                            nn.Conv2d(channels, channels, 1, 1, 0), # Fully-Connected 3
                            act(),
                            nn.Conv2d(channels, channels, 1, 1, 0), # Fully-Connected 4
                            norm_init(channels,h,w),
                            ),
                        split_along_dim=1 # split channelwise
                    ) for _ in range(num_layers)]))

    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x

class SwinReS2MLP(nn.Module):
    def __init__(self, initial_image_size=[64, 64], stages=[3,4,6,3], channels=[16, 32, 64, 128], factor=2, mode='downscale', version=1, activation='gelu', norm='layernorm', padding_mode='replicate', input_channels=None, output_channels=None, input_dim=None, output_dim=None):
        super(SwinReS2MLP, self).__init__()
        if type(stages) == int:
            stages = [stages]
        if type(channels) == int:
            channels = [channels]
        assert mode == 'downscale' or mode == 'upscale', 'mode must be \'upscale\' or \'downscale\''
        if type(initial_image_size) == int:
            h, w = [initial_image_size, initial_image_size]
        else:
            h, w = initial_image_size
        if mode == 'downscale':
            assert all([s % (factor**len(stages)) == 0 for s in [h, w]]), f"initial image size({initial_image_size}) must be able to divide by {factor}^{len(stages)} = {factor**len(stages)}"
        
        # initialize layers
        modules = []
        modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, channels[0]*h*w),
                    nn.Unflatten(1, [channels[0],h,w])
                    ) if input_dim else nn.Identity()) # if given input_dim, start image generation from vector.
        modules.append(nn.Conv2d(input_channels, channels[0], 1, 1, 0) if input_channels else nn.Identity())
        for i, (nlayers, c) in enumerate(zip(stages, channels)):
            modules.append(ReS2MLP2d(c, [h, w], activation=activation, norm=norm, padding_mode=padding_mode, num_layers=nlayers))
            if i != len(stages)-1:
                if mode == 'upscale':
                    modules.append(nn.Upsample(scale_factor=factor))
                    modules.append(nn.Conv2d(channels[i], channels[i+1], 1, 1, 0))
                    h, w = h * factor, w * factor
                if mode == 'downscale':
                    modules.append(nn.AvgPool2d(kernel_size=factor))
                    modules.append(nn.Conv2d(channels[i], channels[i+1], 1, 1, 0))
                    h, w = h // factor, w // factor
        modules.append(nn.Conv2d(channels[-1], output_channels, 1, 1, 0) if output_channels else nn.Identity())
        modules.append(
                nn.Sequential(
                    nn.AvgPool2d(2**(len(stages))),
                    nn.Flatten(),
                    nn.Linear(channels[-1], output_dim) # if given output_dim, added pooling and fully-connected finaly.
                    ) if output_dim else nn.Identity()) 
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


