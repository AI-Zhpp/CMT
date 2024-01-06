import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

        for i in range(depth):
            name = 'Non' + str(i)
            setattr(self, name, nn.Linear(2048, 2048))


    def forward(self, x, mask = None):

        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
class PTE_1(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super(PTE_1, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.conv = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, mask=None):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size)
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_2 = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=16, w=8)
        x_2 = self.conv(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.relu(x_2)
        x_1 = self.to_latent(x_1)
        x_2 = self.to_latent(x_2)

        return x_1, x_2

class PTE_2(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super(PTE_2, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.conv = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size)
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_2 = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=16, w=8)
        x_2 = self.conv(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.relu(x_2)
        x_1 = self.to_latent(x_1)
        x_2 = self.to_latent(x_2)

        return x_1, x_2

class PTE_3(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_3, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.conv = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size)
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_2 = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=16, w=8)
        x_2 = self.conv(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.relu(x_2)
        x_1 = self.to_latent(x_1)
        x_2 = self.to_latent(x_2)

        return x_1, x_2

class PTE_4(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_4, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.conv = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_2 = rearrange(x[:, 2:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=8, w=8)
        x_2 = self.conv(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.relu(x_2)
        x_1 = self.to_latent(x_1)
        x_2 = self.to_latent(x_2)
        return x_1, x_2

class PTE_5(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_5, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.conv = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, mask=None):
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_2 = rearrange(x[:, 2:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=8, w=8)
        x_2 = self.conv(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.relu(x_2)
        x_1 = self.to_latent(x_1)
        x_2 = self.to_latent(x_2)
        return x_1, x_2

class PTE_6(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_6, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.conv = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, mask=None):
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_2 = rearrange(x[:, 2:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=8, w=8)
        x_2 = self.conv(x_2)
        x_2 = self.bn(x_2)
        x_2 = self.relu(x_2)
        x_1 = self.to_latent(x_1)
        x_2 = self.to_latent(x_2)
        return x_1, x_2

class PTE_7(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_7, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
    def forward(self, x, mask=None):
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_1 = self.to_latent(x_1)

        return x_1

class PTE_8(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_8, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
    def forward(self, x, mask=None):
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_1 = self.to_latent(x_1)

        return x_1

class PTE_9(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head,
                 mlp_dim, dropout=0.1):
        super(PTE_9, self).__init__()
        self.p_size = p_size
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
    def forward(self, x, mask=None):
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_1 = x[:, 0]
        x_1 = self.to_latent(x_1)

        return x_1