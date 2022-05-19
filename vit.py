import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Torch implementation of Vision Transformer

# configuration
class CONFIG:
    IMG_SHAPE = 224  # for imagenet


# helper
def pair(n):
    return (n, n) if isinstance(n, int) else n


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer for ViT
    Divide figure into patches, and embed them as a token

    params:
        in_channels:    dimension of figure, default: 3
        patch_size:     size of patch, default: 16
        embed_dim:      dimension of embedding, default: 768
    
    input:
        x:      tensor of shape (batch_size, channels, width, height)
    
    output:
        y:      tensor of shape (batch_size, num_patches**2+1, embed_dim)
    """

    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_shape = pair(patch_size)
        self.num_patches = CONFIG.IMG_SHAPE // self.patch_size
        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(self.num_patches ** 2 + 1, embed_dim))

    def forword(self, x):
        batch_size = x.shape[0]  # (batch_size, embed_dim, num_patches, num_patches)
        x = rearrange(
            self.proj(x), "b e h (w) -> b (h w) e"
        )  # (batch_size, num_patches**2, embed_dim)
        cls_tokens = repeat(
            self.cls_token, "() n e -> b n e", b=batch_size
        )  # (batch_size, 1, embed_dim)
        x = torch.cat(
            [cls_tokens, x], dim=1
        )  # (batch_size, num_patches**2+1, embed_dim)
        x += self.positions

        return x


class MSA(nn.Module):
    """
    Multi-head Self Attention Encoder Block
    From the paper: https://arxiv.org/abs/1706.03762

    params:
        embed_dim:    dimension of embedded tensor
        heads:  number of heads

    input:
        x:      tensor of shape (batch_size, seq_len, embed_dim)

    output:
        y:      tensor of shape (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim, heads):
        super(MSA, self).__init__(self)
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert self.head_dim * self.heads == self.embed_dim
        self.Q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.K_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.V_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.pre_layernorm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        x = self.pre_layernorm1(x)
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        # divide into heads
        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.einsum("bhlq,bhlk->bhll", [Q, K])
        attention = torch.softmax(
            energy / self.embed_dim ** (1 / 2), dim=3
        )  # (batch_size, heads, seq_len, seq_len)
        o = torch.einsum("bhll,bhlv->bhlv", [attention, V]).reshape(
            batch_size, seq_len, dim
        )
        return o


class MLP(nn.Module):
    """
    Multi Layer Perceptron

    params:
        embed_dim:  embed dimension
        p:          droput rate

    input:
        x:          tensor of shape (batch_size, seq_len, embed_dim)
    
    output:
        y:          tensor of shape (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim, p):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        self.out_proj = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GeLU(),
            nn.Dropout(p),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )
        self.pre_layernorm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.pre_layernorm(x)
        x = self.out_proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, heads, p):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.p = p
        self.msa = MSA(embed_dim, heads)
        self.mlp = MLP(embed_dim, p)

    def forward(self, x):
        return self.mlp(self.msa(x))


class ViT(nn.Module):
    def __init__(self, num_layers, embed_dim, heads, p):
        super(ViT, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.heads = heads
        self.p = p
        self.preprocess = PatchEmbedding(embed_dim=self.embed_dim)
        self.backbone = nn.ModuleList(
            [AttentionBlock(embed_dim, heads, p) for _ in range(num_layers)]
        )

        self.classifier = nn.Linear((CONFIG.IMG_SHAPE // 16) ** 2 + 1, 1000)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    dummy_input = torch.zeros(1, 3, 224, 224)
    vit_model = ViT(12, 768, 12, 0.5)
    print(vit_model)
    print(vit_model(dummy_input).shape)
