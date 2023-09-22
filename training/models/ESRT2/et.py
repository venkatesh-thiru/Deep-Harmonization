import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
# LayerNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# TODO: not sure numbers of layer in mlp
class FeedForward(nn.Module):
    def __init__(self, dim, hiddenDim, dropOut=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hiddenDim),
            nn.GELU(),
            nn.Dropout(dropOut),
            nn.Linear(hiddenDim, dim),
            nn.Dropout(dropOut)
        )

    def forward(self, x):
        return self.net(x)

# Efficient Multi-Head Attention
class EMHA(nn.Module):
    def __init__(self, inChannels, splitFactors=4, heads=8):
        super().__init__()
        dimHead = inChannels // (2*heads)

        self.heads = heads
        self.splitFactors = splitFactors
        self.scale = dimHead ** -0.5

        self.reduction = nn.Conv1d(
            in_channels=inChannels, out_channels=inChannels//2, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.toQKV = nn.Linear(
            inChannels // 2, inChannels // 2 * 3, bias=False)
        self.expansion = nn.Conv1d(
            in_channels=inChannels//2, out_channels=inChannels, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(
            self.splitFactors, dim=2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)

        out = torch.cat(tuple(pool), dim=1)
        out = out.transpose(-1, -2)
        out = self.expansion(out)
        return out

class EfficientTransformer(nn.Module):
    def __init__(self, inChannels, mlpDim=256, k=3, splitFactors=4, heads=4, dropOut=0.):
        super().__init__()

        self.k = k
        self.unFold = nn.Unfold(kernel_size=(k, k), padding=k//2)
        self.norm1 = nn.LayerNorm(inChannels*k*k)
        self.emha = EMHA(inChannels=inChannels*k*k,
                         splitFactors=splitFactors, heads=heads)
        self.norm2 = nn.LayerNorm(inChannels*k*k)
        self.mlp = FeedForward(inChannels*k*k, mlpDim, dropOut=dropOut)

    def forward(self, x):
        _, _, h, w = x.shape
        # b c h w -> b (kkc) (hw)
        x = self.unFold(x)
        x = x.transpose(-2, -1)
        x = self.norm1(x)
        x = x.transpose(-2, -1)
        x = self.emha(x) + x
        x = x.transpose(-2, -1)
        x = self.norm2(x)
        x = self.mlp(x) + x
        x = x.transpose(-2, -1)
        return F.fold(x, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))

if __name__ == '__main__':
    # unit test
    # m = EMHA(inChannels = 32)
    # ## B C N
    # x = torch.tensor([float((i+1)%16) for i in range(48*48*32)]).reshape((1, 32, 48*48))
    # y = m(x)
    # print(y.shape)

    # dim = channel // 2, normDim = h*w
    et = EfficientTransformer(inChannels=16, mlpDim=2048)
    # B C H W
    x = torch.tensor([float((i+1) % 16)
                     for i in range(128*128*16)]).reshape((1, 16, 128, 128))
    y = et(x)
    print(y.shape)
