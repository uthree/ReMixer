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
# output: [batch_size, channels, H, W]
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
        self.gelu = nn.GELU()
        self.patch2image = Patch2Image(channels, image_size, patch_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.feature2patch(x)
        x = torch.repeat_interleave(x, self.num_patch, dim=1)
        x = x.reshape(x.shape[0], self.num_patch, self.dim_mixer)
        x = x + self.positional_embedding
        x = self.remixer(x)
        x = self.to_channels(x)
        x = self.gelu(x)
        x = self.patch2image(x)
        x = self.sigmoid(x)
        return x
