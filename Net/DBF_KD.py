import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionBase(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
    ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False
    ):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias
        )

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class LowBranch_of_hr_stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowBranch_of_hr_stem, self).__init__()
        self.conv3x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn_relu1 = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn_relu2 = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn_relu1(x)
        x = self.conv3x3_2(x)
        x = self.bn_relu2(x)
        x = self.conv1x1(x)
        return x


class HighBranch_of_stem(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(HighBranch_of_stem, self).__init__()
        self.dwconv3x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.conv3x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.dwconv3x3(x)
        x = self.bn_relu(x)
        x = self.conv3x3(x)
        return x


class HR_Stem(nn.Module):
    def __init__(self):
        super(HR_Stem, self).__init__()
        self.initial_conv = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.LR = LowBranch_of_hr_stem(32, 32)
        self.HR = HighBranch_of_stem(32, 32)

        self.concat_conv = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn_relu(x)

        lr = self.LR(x)
        hr = self.HR(x)

        combined_features = torch.cat((lr, hr), dim=1)
        return combined_features


class LowBranch_of_hr_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowBranch_of_hr_block, self).__init__()

        # Depthwise Convolution 3x3 with stride=2
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        # Batch Normalization
        self.bn = nn.BatchNorm2d(in_channels)

        # Convolution 1x1
        self.conv1x1_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )

        # HardSwish Activation
        self.hardswish = nn.Hardswish()

        # Convolution 1x1
        self.conv1x1_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, bias=False
        )

        # Upsampling to restore spatial dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.trans_conv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2, bias=False
        )

    def forward(self, x):
        original_h, original_w = x.shape[2], x.shape[3]

        # Apply Depthwise Convolution
        x = self.dwconv(x)

        # Apply Batch Normalization
        x = self.bn(x)

        # Apply 1x1 Convolution
        x = self.conv1x1_1(x)

        # Apply HardSwish Activation
        x = self.hardswish(x)

        # Apply 1x1 Convolution
        x = self.conv1x1_2(x)

        # Upsample to restore original dimensions
        x = self.trans_conv(x)
        # Calculate padding to ensure output dimensions match original dimensions
        padding_h = original_h - x.shape[2]
        padding_w = original_w - x.shape[3]

        # Apply padding
        x = F.pad(x, (0, padding_w, 0, padding_h))
        return x


# class HighBranch_of_hr_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(HighBranch_of_hr_block, self).__init__()

#         # Depthwise Convolution 3x3
#         self.dwconv = nn.Conv2d(
#             in_channels,
#             in_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=in_channels,
#             bias=False,
#         )

#         # Batch Normalization
#         self.bn = nn.BatchNorm2d(in_channels)

#         # Convolution 1x1
#         self.conv1x1_1 = nn.Conv2d(
#             in_channels, out_channels, kernel_size=1, stride=1, bias=False
#         )

#         # HardSwish Activation
#         self.hardswish = nn.Hardswish()

#         # Convolution 1x1
#         self.conv1x1_2 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=1, stride=1, bias=False
#         )

#     def forward(self, x):
#         # Apply Depthwise Convolution
#         x = self.dwconv(x)

#         # Apply Batch Normalization
#         x = self.bn(x)

#         # Apply 1x1 Convolution
#         x = self.conv1x1_1(x)

#         # Apply HardSwish Activation
#         x = self.hardswish(x)

#         # Apply 1x1 Convolution
#         x = self.conv1x1_2(x)

#         return x


class HighBranch_of_hr_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HighBranch_of_hr_block, self).__init__()

        # Depthwise Convolution 3x3
        self.dwconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        # Batch Normalization
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Apply Depthwise Convolution
        x = self.dwconv(x)

        # Apply Batch Normalization
        x = self.bn(x)

        return x


class HR_Block(nn.Module):
    def __init__(self):
        super(HR_Block, self).__init__()
        self.LR = LowBranch_of_hr_block(64, 64)
        self.HR = HighBranch_of_hr_block(64, 64)

    def forward(self, x):
        lr = self.LR(x)
        hr = self.HR(x)

        combined_features = lr + hr
        return combined_features + x


class BaseFeatureExtraction(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=1.0,
        qkv_bias=False,
    ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, "WithBias")
        self.attn = AttentionBase(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, "WithBias")
        self.mlp = Mlp(
            in_features=dim,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, x):
        # [8, 64, 128, 128]
        norm = self.norm1(x)
        x = x + self.attn(norm)
        norm2 = self.norm2(x)
        x = x + self.mlp(norm2)
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(
            64, 64, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


import numbers


## Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(1)
            ],
        )

        self.deatail_hr_block = nn.Sequential(
            *[HR_Block() for i in range(3)],
        )
        self.base_transformer_pre = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(2)
            ],
        )

        self.baseFeatureExtractor = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeatureExtractor = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        base_pre = self.base_transformer_pre(out_enc_level1)
        deatil_pre = self.deatail_hr_block(out_enc_level1)

        base_feature = self.baseFeatureExtractor(base_pre)
        detail_feature = self.detailFeatureExtractor(deatil_pre)

        return base_feature, detail_feature, out_enc_level1


class Restormer_Encoder2(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Encoder2, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_low = nn.Sequential(
            HR_Stem(),
            LowBranch_of_hr_block(64, 64),
            DetailFeatureExtraction(num_layers=1),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(4)
            ],
        )

        self.encoder_high = nn.Sequential(
            HR_Stem(),
            HighBranch_of_hr_block(64, 64),
            DetailFeatureExtraction(num_layers=1),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(4)
            ],
        )

        self.encoder_base = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.encoder_detail = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        low_features = self.encoder_low(inp_enc_level1)
        high_features = self.encoder_high(inp_enc_level1)

        high_base_feature = self.encoder_base(high_features)
        low_base_feature = self.encoder_base(low_features)

        hight_detail_feature = self.encoder_detail(high_features)
        low_detail_feature = self.encoder_detail(low_features)

        return (
            high_base_feature,
            hight_detail_feature,
            low_base_feature,
            low_detail_feature,
        )


class Restormer_Decoder(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(
            int(dim * 2), int(dim), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


class Restormer_Decoder2(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Decoder2, self).__init__()
        self.reduce_channel = nn.Conv2d(
            int(dim * 2), int(dim), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, bh_feature, bl_feature, dh_feature, dl_feature):
        out_enc_level0_L = torch.cat((bl_feature, dl_feature), dim=1)
        out_enc_level0_H = torch.cat((bh_feature, dh_feature), dim=1)

        out_enc_level0_L = self.reduce_channel(out_enc_level0_L)
        out_enc_level0_H = self.reduce_channel(out_enc_level0_H)

        out_enc_level0 = torch.cat((out_enc_level0_L, out_enc_level0_H), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)

        out_enc_level1 = self.encoder_level2(out_enc_level0)

        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels):
        super(AffineCouplingLayer, self).__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        s, t = self.net(x1).chunk(2, 1)
        x2 = x2 * torch.exp(s) + t
        return torch.cat([x1, x2], 1)

    def inverse(self, x):
        x1, x2 = x.chunk(2, 1)
        s, t = self.net(x1).chunk(2, 1)
        x2 = (x2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], 1)


class Invertible1x1Conv(nn.Module):
    def __init__(self, in_channels):
        super(Invertible1x1Conv, self).__init__()
        weight = torch.qr(torch.randn(in_channels, in_channels))[0]
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return F.conv2d(x, self.weight.unsqueeze(-1).unsqueeze(-1))

    def inverse(self, x):
        return F.conv2d(x, torch.linalg.inv(self.weight).unsqueeze(-1).unsqueeze(-1))


class Glow(nn.Module):
    def __init__(self, in_channels, num_blocks=4):
        super(Glow, self).__init__()
        self.invertible1x1convs = nn.ModuleList(
            [Invertible1x1Conv(in_channels) for _ in range(num_blocks)]
        )
        self.affine_coupling_layers = nn.ModuleList(
            [AffineCouplingLayer(in_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for conv, coupling in zip(self.invertible1x1convs, self.affine_coupling_layers):
            x = conv(x)
            x = coupling(x)
        return x

    def inverse(self, x):
        for conv, coupling in reversed(
            list(zip(self.invertible1x1convs, self.affine_coupling_layers))
        ):
            x = coupling.inverse(x)
            x = conv.inverse(x)
        return x


class DetailNode_Glow(nn.Module):
    def __init__(self, inp=32, oup=32, expand_ratio=2):
        super(DetailNode_Glow, self).__init__()
        self.theta_phi = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.theta_rho = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.theta_eta = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.glow = Glow(inp * 2)
        self.shuffle_conv = nn.Conv2d(
            2 * inp, 2 * inp, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        assert x.shape[1] % 2 == 0, "Input channels must be divisible by 2."
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]
        return z1, z2

    def forward(self, z1, z2):
        combined = torch.cat((z1, z2), dim=1)
        shuffled = self.shuffle_conv(combined)
        glow_out = self.glow(shuffled)
        z1, z2 = self.separateFeature(glow_out)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class FlowLayer(nn.Module):
    def __init__(self, input_dim):
        super(FlowLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(input_dim)
        self.affine = nn.Conv2d(input_dim, input_dim * 2, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        s, t = self.affine(h).chunk(2, dim=1)
        s = torch.clamp(s, min=-10, max=10)
        return x * torch.exp(s) + t


class FlowPlusPlus(nn.Module):
    def __init__(self, input_dim, num_layers=5):
        super(FlowPlusPlus, self).__init__()
        self.layers = nn.ModuleList([FlowLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DetailNode_FlowPlus(nn.Module):
    def __init__(self, inp=32, oup=32, expand_ratio=2):
        super(DetailNode_FlowPlus, self).__init__()
        self.theta_phi = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.theta_rho = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.theta_eta = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )

        self.flow = FlowPlusPlus(input_dim=2 * inp)
        self.shuffle_conv = nn.Conv2d(
            2 * inp, 2 * inp, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]
        return z1, z2

    def forward(self, z1, z2):
        combined = torch.cat((z1, z2), dim=1)
        shuffled = self.shuffle_conv(combined)
        flow_out = self.flow(shuffled)
        z1, z2 = self.separateFeature(flow_out)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class IAF(nn.Module):
    def __init__(self, in_channels):
        super(IAF, self).__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2")
        # Define a small MLP to parameterize the shift and scale functions
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Split input into two halves
        x1, x2 = x.chunk(2, dim=1)
        # Compute the shift and scale parameters
        s, t = self.net(x1).chunk(2, dim=1)
        # Apply the affine transformation
        x2 = x2 * torch.exp(s) + t
        return torch.cat([x1, x2], dim=1)

    def inverse(self, x):
        # Split input into two halves
        x1, x2 = x.chunk(2, dim=1)
        # Compute the shift and scale parameters
        s, t = self.net(x1).chunk(2, dim=1)
        # Apply the inverse affine transformation
        x2 = (x2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=1)


class DetailNode_IAF(nn.Module):
    def __init__(self, inp=32, oup=32, expand_ratio=2):
        super(DetailNode_IAF, self).__init__()
        # Initialize InvertedResidualBlocks
        self.theta_phi = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.theta_rho = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )
        self.theta_eta = InvertedResidualBlock(
            inp=inp, oup=oup, expand_ratio=expand_ratio
        )

        # Initialize IAF module
        self.iaf = IAF(inp * 2)

        # Shuffle convolution
        self.shuffle_conv = nn.Conv2d(
            2 * inp, 2 * inp, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        # Ensure the input tensor x can be split evenly
        assert x.shape[1] % 2 == 0, "Input channels must be divisible by 2."
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]
        return z1, z2

    def forward(self, z1, z2):
        # Concatenate z1 and z2 and apply shuffle convolution
        combined = torch.cat((z1, z2), dim=1)
        shuffled = self.shuffle_conv(combined)

        # Apply IAF
        iaf_out = self.iaf(shuffled)

        # Separate features after IAF
        z1, z2 = self.separateFeature(iaf_out)

        # Apply transformations and compute final outputs
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)

        return z1, z2
