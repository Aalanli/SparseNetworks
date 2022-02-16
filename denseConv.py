# %%
import torch
import torch.nn.functional as F

from sparseConv import *


class DenseConvNode(nn.Module):
    def __init__(self, comp, col) -> None:
        super().__init__()
        self.comp = copy.deepcopy(comp).apply(reset_parameters)
        self.col = copy.deepcopy(col).apply(reset_parameters)
    
    def forward(self, x: List[torch.Tensor]):
        #batch = x[0].shape[0]
        x = torch.cat(x, dim=1) # [batch * nodes, ...]
        #x = self.comp(x) # [batch * nodes, ...]
        #x = x.reshape([-1, batch] + list(x.shape[1:])).sum(0) # [batch, ...]
        return self.col(x)

class DenseConvLayer(nn.Module):
    def __init__(self, nodes, comp, col) -> None:
        super().__init__()
        self.nodes = nn.ModuleList([DenseConvNode(comp, col) for i in range(nodes)])
    
    def forward(self, x):
        nodes = []
        for layer in self.nodes:
            nodes.append(layer(x))
        return nodes

class DenseConvNet(nn.Module):
    def __init__(self, split_patch, dim, layers, patch_size, comp, col, n_classes) -> None:
        super().__init__()
        split_nodes = split_patch ** 2
        self.split_patch = split_patch
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim))
        self.layers = nn.ModuleList([DenseConvLayer(split_nodes, comp, col) for i in range(layers)])
        self.out_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes))
        
    def forward(self, x):
        x = self.patch_embed(x)
        pad_x = int(x.shape[3] / self.split_patch + 0.5) * self.split_patch - x.shape[3]
        pad_y = int(x.shape[2] / self.split_patch + 0.5) * self.split_patch - x.shape[2]
        x = F.pad(x, (0, pad_x, 0, pad_y))
        patches = map(lambda x: list(torch.chunk(x, self.split_patch, -2)), list(torch.chunk(x, self.split_patch, -1)))
        x0 = [j for i in patches for j in i]
        for layer in self.layers:
            x0 = layer(x0)
        out = [torch.cat(x0[i:i+self.split_patch], dim=-2) for i in range(0, len(x0), self.split_patch)]
        out = torch.cat(out, dim=-1)
        return self.out_proj(out)

"""comp = nn.Conv2d(64 * 9, 256, 1, padding='same')
col = dense_conv(256, 64)
f = DenseConvNet(3, 64, 6, 9, comp, col, 62).cuda()
x = torch.rand(4, 3, 512, 512).cuda()
y = f(x)
print(y.shape)"""

