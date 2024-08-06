"""

1. ~~Patchify latents~~
2. ~~Learnable positional embeddings~~
    * Sum of space and time position embeddings
3. ~~Positional embeddings are summed with linear projections~~
4. ~~Spatial window and spatiotemporal window attention modules~~
5. Encoder blocks and full transformer
6. Condition using autoregressive generation (4.4) and optionally self-conditioning
    * Since text conditioning isn't being done, determine what to do instead of spatial cross-attention
    * Cross attention distinguishes itself by receiving 2 sequences instead of 1. It does this by also receiving the sequence processed to date by the decoder.
    * This is sufficiently general purpose that it's worth keeping for the actions data.
    * Ada-LN probably not needed here since it's better for smooth, global changes, which actions data will not produce.

"""
import torch
import torch.nn as nn


class EmbeddingProjection(nn.Module):
    def __init__(self, image_size, frames, in_channels, hidden_dim, patch_size, dropout):
        super().__init__()
        self.image_size = image_size
        self.frames = frames
        self.patch_size = patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.embed = nn.Conv3d(in_channels, hidden_dim, (1, patch_size, patch_size), (1, patch_size, patch_size))
        self.temporal_pos_emb = nn.Parameter(torch.ones(frames, hidden_dim))
        self.spatial_pos_emb = nn.Parameter(torch.ones(self.num_patches, self.hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(-1, self.num_patches * self.frames, self.hidden_dim)
        temporal_pos_emb = self.temporal_pos_emb.unsqueeze(1).expand(-1, self.num_patches, -1)
        spatial_pos_emb = self.spatial_pos_emb.unsqueeze(0).expand(self.frames, -1, -1)
        positional_embeddings = temporal_pos_emb + spatial_pos_emb
        positional_embeddings = positional_embeddings.view(1, self.frames * self.num_patches, self.hidden_dim)
        x = x + positional_embeddings
        x = self.dropout(x)
        return x


class SpatiotemporalAttention(nn.Module):
    def __init__(self, image_size, hidden_dim, query_dim, context_dim, window_size: tuple[int, int, int], frames,
                 patch_size, num_heads, dropout):
        super().__init__()
        context_dim = context_dim if context_dim else query_dim

        self.window_size = window_size
        self.num_heads = num_heads
        self.frames = frames
        self.patch_dim = (image_size // patch_size)
        self.head_dim = hidden_dim // num_heads
        # self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=False)
        self.q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, condition=None):
        B, N, C = x.shape
        T, H, W = self.window_size

        x = x.view(B, self.frames, self.patch_dim, self.patch_dim, C)  # [batch_dim, frames, patch_size, patch_size, channel]
        x = x.unfold(1, T, T).unfold(2, H, H).unfold(3, W, W)
        B, num_windows_T, num_windows_H, num_windows_W, C, T, H, W = x.shape

        x = x.permute(0, 1, 2, 3, 5, 6, 7, 4).contiguous()
        x = x.view(B, num_windows_T * num_windows_H * num_windows_W, T * H * W, C)

        context = condition if condition else x

        # Apply attention within each window
        q = self.q(x).reshape(B, num_windows_T * num_windows_H * num_windows_W, T * H * W, self.num_heads, self.head_dim)
        q = q.permute(0, 3, 1, 2, 4)

        k = self.k(context).reshape(B, num_windows_T * num_windows_H * num_windows_W, T * H * W, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4)

        v = self.v(context).reshape(B, num_windows_T * num_windows_H * num_windows_W, T * H * W, self.num_heads, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attn = attn.softmax(dim=-1)

        windows = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, num_windows_T * num_windows_H * num_windows_W, T * H * W, C)
        windows = self.out(windows)
        windows = self.drop(windows)

        # Reconstruct the 3D structure from windows
        x = windows.view(B, num_windows_T, num_windows_H, num_windows_W, T, H, W, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, self.frames, self.patch_dim, self.patch_dim, C)

        # Flatten back to 1D string of patches
        x = x.view(B, N, C)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, image_size, hidden_dim, query_dim, context_dim, frames, patch_size, num_heads, dropout):
        super().__init__()
        context_dim = context_dim if context_dim else query_dim

        self.num_heads = num_heads
        self.frames = frames
        self.patch_dim = (image_size // patch_size)
        self.head_dim = hidden_dim // num_heads
        # self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=False)
        self.q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, condition=None):
        B, N, C = x.shape

        x = x.view(B, self.frames, self.patch_dim, self.patch_dim, C)  # [batch_dim, frames, patch_size, patch_size, channel]

        x = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        B, T, num_windows_H, num_windows_W, C, H, W = x.shape

        x = x.permute(0, 1, 2, 3, 5, 6, 4).contiguous() # [B, T, num_windows_H, num_windows_W, H, W, C]
        x = x.view(B, T * num_windows_H * num_windows_W, H * W, C)

        context = condition if condition else x


        # Apply attention within each window
        q = self.q(x).reshape(B, T * num_windows_H * num_windows_W, H * W, self.num_heads, self.head_dim)
        q = q.permute(0, 3, 1, 2, 4)

        k = self.k(context).reshape(B, T * num_windows_H * num_windows_W, H * W, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4)

        v = self.v(context).reshape(B, T * num_windows_H * num_windows_W, H * W, self.num_heads, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, T * num_windows_H * num_windows_W, H * W, C)
        x = self.out(x)
        x = self.drop(x)

        # Reconstruct the 3D structure from windows
        x = x.view(B, T, num_windows_H, num_windows_W, H, W, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

        x = x.view(B, self.frames, self.patch_dim, self.patch_dim, C)

        # Flatten back to 1D string of patches
        x = x.view(B, N, C)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, image_size, hidden_dim, context_dim, frames, patch_size, window_size, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.local_attention = SpatialAttention(image_size, hidden_dim, hidden_dim, None, frames, patch_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.local_cross_attention = SpatialAttention(image_size, hidden_dim, hidden_dim, context_dim, frames, patch_size, num_heads, dropout)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.global_attention = SpatiotemporalAttention(image_size, hidden_dim, hidden_dim, None, window_size, frames, patch_size, num_heads, dropout)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.global_cross_attention = SpatiotemporalAttention(image_size, hidden_dim, hidden_dim, context_dim, window_size, frames, patch_size, num_heads, dropout)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.feedforward = MLP(hidden_dim, hidden_dim, dropout)

    def forward(self, x, context=None):
        # Spatial Self-Attention
        x_spatial_norm = self.ln1(x)
        x_spatial_att = self.local_attention(x_spatial_norm)
        x_spatial = x + x_spatial_att

        # Spatial Cross-Attention
        x_spatial_cross_norm = self.ln2(x_spatial)
        x_spatial_cross_att = self.local_cross_attention(x_spatial_cross_norm, context)
        x_spatial_cross = x_spatial + x_spatial_cross_att

        # Spatiotemporal Self-Attention
        x_spatiotemporal_norm = self.ln3(x_spatial_cross)
        x_spatiotemporal_att = self.global_attention(x_spatiotemporal_norm)
        x_spatiotemporal = x_spatial_cross + x_spatiotemporal_att

        # Spatiotemporal Cross-Attention
        x_spatiotemporal_cross_norm = self.ln4(x_spatiotemporal)
        x_spatiotemporal_cross_att = self.global_cross_attention(x_spatiotemporal_cross_norm, context)
        x_spatiotemporal_cross = x_spatiotemporal + x_spatiotemporal_cross_att

        # Feedforward Network
        x_ff_norm = self.ln5(x_spatiotemporal_cross)
        x_ff = self.feedforward(x_ff_norm)
        x_output = x_spatiotemporal_cross + x_ff

        return x_output

class DiT(nn.Module):
    def __init__(self, n_layers, image_size, hidden_dim, query_dim, context_dim, frames, patch_size, window_size, num_heads, dropout):
        super().__init__()
        self.patch_embed = EmbeddingProjection(image_size, frames, query_dim, hidden_dim, patch_size, dropout)
        self.transformer = nn.ModuleList([EncoderBlock(image_size, hidden_dim, context_dim, frames, patch_size, window_size, num_heads, dropout) for _ in range(n_layers)])

    def forward(self, x):
        x = self.patch_embed(x)
        for enc_block in self.transformer:
            x = enc_block(x)
        return x

if __name__ == "__main__":
    # [16, 3, 128, 128] -> [1, 256, 4, 16, 16] (note: 18 is the z_channel / encoder_out dimension)
    t = torch.randn(3, 256, 4, 16, 16)
    model = DiT(6, 16, 256, 256, None, 4, 1, (4, 4, 4), 8, 0.1)
    print(sum([p.numel() for p in model.parameters()]))
    # print(model(t).shape)
