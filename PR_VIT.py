import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

@dataclass(frozen=True)
class VarGroup:
    """Defines a variable (or variable-group) by the channel indices it uses from X."""
    name: str
    channels: List[int]   # indices into the C dimension of X: [B, C, H, W]

class PatchEmbed(nn.Module):
    """
    Patchify + linear projection via Conv2d.
    Input : [B, Cin, H, W]
    Output: [B, Np, D] where Np=(H/Py)*(W/Px)
    """
    def __init__(self, in_ch: int, patch_size: Tuple[int, int], embed_dim: int):
        super().__init__()
        py, px = patch_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(py, px), stride=(py, px), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Cin, H, W]
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, Np, D]: this is the sequence: Np = number of patches (secquence length). D = embedding size.
        return x
class CrossAttentionGate(nn.Module):
    """
    Single cross-attention gate:
    out, attn = MHA(query, key, value)
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,   # [B, 1, D] or [B, Nq, D]
        kv: torch.Tensor,  # [B, Nk, D]
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm (stabilizes training)
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        out, attn = self.mha(query=qn, key=kvn, value=kvn, need_weights=need_weights)
        out = q + self.dropout(out)  # residual
        return out, attn

class PRViT(nn.Module):
    """
    Variable-aware "ViT-like" model with TWO attention gates:
      Gate 1: variable-token -> spatial patches (per variable/group)
      Gate 2: PR-token       -> variable summaries

    Inputs:
      X: [B, C, H, W]  (3-hourly / 6-hourly maps; one timestamp per sample)

    Output:
      y_hat: [B, out_dim]
      plus attention maps for interpretability.

    Notes:
      - Variable tokens and PR token are LEARNED parameters (not data inputs).
      - Patchification is done PER variable/group (clean variable interpretability).
    """
    def __init__(
        self,
        var_groups: List[VarGroup],
        in_channels_total: int,                 # C (for basic validation)
        patch_size: Tuple[int, int] = (8, 8),    # (Py, Px)
        embed_dim: int = 256,                   # D
        num_heads_space: int = 1,
        num_heads_vars: int = 1,
        dropout: float = 0.0,
        out_dim: int = 5,                       # e.g., AOD vector length (wavelengths) or 1 for scalar
        use_aeronet_at_t: bool = False,         # if you want to concatenate AERONET at time t
        aeronet_dim: int = 0,                   # set to L if use_aeronet_at_t=True
    ):
        super().__init__()
        assert len(var_groups) > 0, "Need at least one VarGroup."
        self.var_groups = var_groups
        self.Nv = len(var_groups)
        self.C = in_channels_total
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_aeronet_at_t = use_aeronet_at_t
        self.aeronet_dim = aeronet_dim

        # One PatchEmbed per variable/group
        self.patch_embeds = nn.ModuleDict()
        for g in var_groups:
            self.patch_embeds[g.name] = PatchEmbed(
                in_ch=len(g.channels),
                patch_size=patch_size,
                embed_dim=embed_dim
            )

        # Learnable tokens:
        # variable query tokens: [Nv, D]
        self.var_tokens = nn.Parameter(torch.randn(self.Nv, embed_dim) * 0.02)
        # PR query token: [1, D]
        self.pr_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Positional embedding for patches: we create it lazily once we know H', W'
        self.pos_embed: Optional[nn.Parameter] = None
        self.pos_shape: Optional[Tuple[int, int]] = None  # (Hp, Wp) in patch grid

        # Two attention gates
        self.var_to_space = CrossAttentionGate(embed_dim, num_heads=num_heads_space, dropout=dropout)
        self.pr_to_vars = CrossAttentionGate(embed_dim, num_heads=num_heads_vars, dropout=dropout)

        # Regression head
        head_in = embed_dim + (aeronet_dim if use_aeronet_at_t else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in, out_dim),
        )

    def _get_pos_embed(self, Hp: int, Wp: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Create (or reuse) learnable positional embedding for patch tokens:
          pos: [1, Np, D] where Np = Hp*Wp
        """
        if (self.pos_embed is None) or (self.pos_shape != (Hp, Wp)):
            Np = Hp * Wp
            self.pos_embed = nn.Parameter(torch.randn(1, Np, self.embed_dim) * 0.02)
            self.pos_shape = (Hp, Wp)
        return self.pos_embed.to(device=device, dtype=dtype)

    def forward(
        self,
        X: torch.Tensor,                 # [B, C, H, W]
        aeronet_t: Optional[torch.Tensor] = None,   # [B, aeronet_dim] if used
        return_attn: bool = True
    ):
        """
        Returns:
          y_hat: [B, out_dim]
          attn_space: Dict[var_name] -> [B, 1, Np] (or [B, num_heads, 1, Np] depending on torch version)
          attn_vars:  [B, 1, Nv] (or head-expanded)
        """
        if X.ndim != 4:
            raise ValueError(f"X must be [B,C,H,W], got {tuple(X.shape)}")
        B, C, H, W = X.shape
        if C != self.C:
            raise ValueError(f"Expected C={self.C}, got C={C}")

        py, px = self.patch_size
        if (H % py) != 0 or (W % px) != 0:
            raise ValueError(f"H,W must be divisible by patch_size {self.patch_size}. Got H={H}, W={W}.")

        Hp, Wp = H // py, W // px
        Np = Hp * Wp

        # Positional embedding for patch tokens
        pos = self._get_pos_embed(Hp, Wp, device=X.device, dtype=X.dtype)  # [1, Np, D]

        # ---- Gate #1: Variable -> Space (per variable/group) ----
        var_summaries = []
        attn_space: Dict[str, torch.Tensor] = {}

        for i, g in enumerate(self.var_groups):
            # select channels for this variable/group
            Xg = X[:, g.channels, :, :]  # [B, Cg, H, W]

            ###### patchify+embed -> [B, Np, D]#########
            Pg = self.patch_embeds[g.name](Xg)
            Pg = Pg + pos  # add 2D pos embed (flattened)

            # variable query token -> [B, 1, D]
            vg = self.var_tokens[i].view(1, 1, -1).expand(B, 1, -1)

            # cross-attn: variable token queries its own patches
            sg, a_sp = self.var_to_space(vg, Pg, need_weights=return_attn)  # sg: [B,1,D]
            var_summaries.append(sg)

            if return_attn and (a_sp is not None):
                # a_sp typically [B, 1, Np] (averaged over heads by default)
                attn_space[g.name] = a_sp

        # Stack summaries: [B, Nv, D]
        S = torch.cat(var_summaries, dim=1)

        # ---- Gate #2: PR -> Variables ----
        pr = self.pr_token.view(1, 1, -1).expand(B, 1, -1)  # [B,1,D]
        z, attn_vars = self.pr_to_vars(pr, S, need_weights=return_attn)  # z: [B,1,D]

        # ---- Regression head ----
        z0 = z[:, 0, :]  # [B, D]
        if self.use_aeronet_at_t:
            if aeronet_t is None:
                raise ValueError("use_aeronet_at_t=True but aeronet_t=None")
            if aeronet_t.shape != (B, self.aeronet_dim):
                raise ValueError(f"aeronet_t must be [B,{self.aeronet_dim}], got {tuple(aeronet_t.shape)}")
            z0 = torch.cat([z0, aeronet_t.to(z0.dtype)], dim=1)

        y_hat = self.head(z0)  # [B, out_dim]

        if return_attn:
            # Also return patch-grid shape so you can reshape attention to [Hp, Wp]
            meta = {"Hp": Hp, "Wp": Wp, "Np": Np, "var_names": [g.name for g in self.var_groups]}
            return y_hat, attn_space, attn_vars, meta
        return y_hat
