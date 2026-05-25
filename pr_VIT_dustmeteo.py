from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn


# --------------------------------------------------
# 3D patch embedding for route meteo branch
# No temporal patching if patch_size=(1, py, px)
# --------------------------------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, patch_size=(1, 20, 20), embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        pt, py, px = patch_size
        self.proj = nn.Conv3d(
            in_channels=in_ch,
            out_channels=embed_dim,
            kernel_size=(pt, py, px),
            stride=(pt, py, px),
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        z = self.proj(x)                    # [B, D, Tp, Hp, Wp]
        B, D, Tp, Hp, Wp = z.shape
        z = z.flatten(2).transpose(1, 2)   # [B, Nt, D]
        return z, (Tp, Hp, Wp)


# --------------------------------------------------
# Cross-attention gate
# --------------------------------------------------
class CrossAttentionGate(nn.Module):
    def __init__(self, embed_dim: int, num_heads=1, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, q, kv, need_weights=True):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        out, attn = self.attn(
            query=qn,
            key=kvn,
            value=kvn,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        out = out + q
        out = out + self.mlp(self.out_norm(out))
        return out, attn


# --------------------------------------------------
# Simple Africa dust encoder (Option A)
# Input:  [B, T, 2, H, W]
# Output: [B, D]
# --------------------------------------------------
class DustSourceEncoder(nn.Module):
    def __init__(self, in_channels=2, embed_dim=256, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(128),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, T, C, H, W] -> Conv3d expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # [B, 2, T, H, W]
        z = self.net(x)                             # [B, 128, T, H, W]
        z = self.pool(z)                            # [B, 128, 1, 1, 1]
        z = self.proj(z)                            # [B, D]
        return z


# --------------------------------------------------
# Route meteo encoder with per-channel space-time attention
# Input:  [B, T, C_r, H, W]
# Output: [B, C_r, D] + attn maps per channel
# --------------------------------------------------
class RouteMeteoEncoder(nn.Module):
    def __init__(
        self,
        channel_names: List[str],
        patch_size=(1, 20, 20),
        embed_dim=256,
        num_heads_space=1,
        dropout=0.0,
    ):
        super().__init__()
        self.channel_names = channel_names
        self.Nc = len(channel_names)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pt, self.py, self.px = patch_size

        self.patch_embeds = nn.ModuleDict({
            ch: PatchEmbed3D(in_ch=1, patch_size=patch_size, embed_dim=embed_dim)
            for ch in channel_names
        })

        self.channel_tokens = nn.Parameter(torch.randn(self.Nc, embed_dim) * 0.02)

        self.channel_to_spacetime = CrossAttentionGate(
            embed_dim=embed_dim,
            num_heads=num_heads_space,
            dropout=dropout,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        self.pos_shape: Optional[Tuple[int, int, int]] = None

    def _get_pos_embed(self, Tp, Hp, Wp, device, dtype):
        if (self.pos_embed is None) or (self.pos_shape != (Tp, Hp, Wp)):
            Nt = Tp * Hp * Wp
            self.pos_embed = nn.Parameter(torch.randn(1, Nt, self.embed_dim) * 0.02)
            self.pos_shape = (Tp, Hp, Wp)
        return self.pos_embed.to(device=device, dtype=dtype)

    def forward(self, X, return_attn=True):
        # X: [B, T, C, H, W]
        B, T, C, H, W = X.shape
        if C != self.Nc:
            raise ValueError(f"Expected {self.Nc} route channels, got {C}")

        if (T % self.pt) != 0 or (H % self.py) != 0 or (W % self.px) != 0:
            raise ValueError("Route input dims must be divisible by patch_size.")

        Tp = T // self.pt
        Hp = H // self.py
        Wp = W // self.px
        pos = self._get_pos_embed(Tp, Hp, Wp, X.device, X.dtype)

        summaries = []
        attn_maps: Dict[str, torch.Tensor] = {}

        for i, ch_name in enumerate(self.channel_names):
            Xc = X[:, :, i:i+1, :, :]                    # [B,T,1,H,W]
            Xc = Xc.permute(0, 2, 1, 3, 4).contiguous() # [B,1,T,H,W]

            Pc, (Tp_, Hp_, Wp_) = self.patch_embeds[ch_name](Xc)
            Pc = Pc + pos

            qc = self.channel_tokens[i].view(1, 1, -1).expand(B, 1, -1)
            sc, a = self.channel_to_spacetime(qc, Pc, need_weights=return_attn)

            summaries.append(sc)

            if return_attn and a is not None:
                attn_maps[ch_name] = a.view(B, 1, Tp, Hp, Wp)

        S = torch.cat(summaries, dim=1)   # [B, C_r, D]
        meta = {"Tp": Tp, "Hp": Hp, "Wp": Wp}
        return S, attn_maps, meta


# --------------------------------------------------
# Full model: Africa dust + route meteo
# --------------------------------------------------
class PRAfricaDustRouteMeteoNet(nn.Module):
    def __init__(
        self,
        route_channel_names: List[str],
        route_patch_size=(1, 20, 20),
        embed_dim=256,
        num_heads_space=1,
        num_heads_fusion=1,
        dropout=0.0,
        out_dim=1,
        use_aeronet_at_t=False,
        aeronet_dim=0,
    ):
        super().__init__()

        self.use_aeronet_at_t = use_aeronet_at_t
        self.aeronet_dim = aeronet_dim

        self.dust_encoder = DustSourceEncoder(
            in_channels=2,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.route_encoder = RouteMeteoEncoder(
            channel_names=route_channel_names,
            patch_size=route_patch_size,
            embed_dim=embed_dim,
            num_heads_space=num_heads_space,
            dropout=dropout,
        )

        self.pr_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.pr_to_fused = CrossAttentionGate(
            embed_dim=embed_dim,
            num_heads=num_heads_fusion,
            dropout=dropout,
        )

        head_in = embed_dim + (aeronet_dim if use_aeronet_at_t else 0)

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in, out_dim),
        )

    def forward(
        self,
        X_dust,                      # [B, T, 2, H_d, W_d]
        X_route,                     # [B, T, C_r, H_r, W_r]
        aeronet_t=None,              # optional [B, aeronet_dim]
        return_attn=True,
    ):
        B = X_dust.shape[0]

        # Dust source summary
        z_dust = self.dust_encoder(X_dust)           # [B, D]
        z_dust_tok = z_dust.unsqueeze(1)             # [B, 1, D]

        # Route summaries + route attention maps
        S_route, attn_route, meta_route = self.route_encoder(
            X_route, return_attn=return_attn
        )                                            # [B, C_r, D]

        # Fuse dust source with route summaries
        S_all = torch.cat([z_dust_tok, S_route], dim=1)   # [B, 1 + C_r, D]

        pr = self.pr_token.view(1, 1, -1).expand(B, 1, -1)
        z, attn_fusion = self.pr_to_fused(pr, S_all, need_weights=return_attn)

        z0 = z[:, 0, :]   # [B, D]

        if self.use_aeronet_at_t:
            if aeronet_t is None:
                raise ValueError("use_aeronet_at_t=True but aeronet_t=None")
            if aeronet_t.shape != (B, self.aeronet_dim):
                raise ValueError(
                    f"aeronet_t must be [B,{self.aeronet_dim}], got {tuple(aeronet_t.shape)}"
                )
            z0 = torch.cat([z0, aeronet_t.to(z0.dtype)], dim=1)

        y_hat = self.head(z0)

        if return_attn:
            meta = {
                "route_channels": self.route_encoder.channel_names,
                "meta_route": meta_route,
                "fusion_token_names": ["dust_source"] + self.route_encoder.channel_names,
            }
            return y_hat, attn_route, attn_fusion, meta

        return y_hat