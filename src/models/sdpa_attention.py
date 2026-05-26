"""SDPA-based MultiheadAttention drop-in replacement.

``torch.nn.functional.scaled_dot_product_attention`` (SDPA) auto-routes
to FlashAttention-2 on Ampere+ GPUs, the memory-efficient kernel on
older GPUs, and the math fallback on CPU. On any of those paths it's
faster AND uses O(N) memory for the attention matrix (instead of
O(N²) like a naive softmax implementation), which is what makes
exact-softmax attention tractable on the 30-80k cell slices that
appear in scgg/LUNA workloads.

PyTorch's stock ``nn.MultiheadAttention`` *does* use SDPA internally
in 2.x, but only when a finicky set of conditions hold (correct mask
format, batch_first, need_weights=False, no nested tensors, no
additive attn_mask combined with key_padding_mask, ...). When any
condition fails it silently falls back to the O(N²)-memory native
path, which is exactly the trap we want to avoid for big slices.

This module exposes ``SDPAMultiheadAttention`` — a thin, explicit
SDPA wrapper with the *same forward signature* as
``nn.MultiheadAttention(batch_first=True, need_weights=False)`` so it
drops in 1-for-1 at every call site in dit_backbone / perceiver_
backbone / latent_vae / latent_diffusion_wrapper.

Key behaviours:
  * Padding via ``key_padding_mask`` (True at PAD, the
    ``nn.MultiheadAttention`` convention) is converted to the SDPA
    additive bias form (True at ALLOWED).
  * Separate q/kv stream supported via ``kdim`` / ``vdim`` so the
    cross-attention blocks in Perceiver work unchanged.
  * Returns ``(out, None)`` to match the (output, weights) tuple
    contract of ``nn.MultiheadAttention``. The second element is
    always ``None`` because exposing attention weights would force
    materialising the N×N matrix and lose the memory-efficient path.
  * Dropout applied inside SDPA when ``self.training`` (matches MHA).
  * Bias on the qkv projections is on by default (matches MHA).

Sequence-length safety: SDPA *itself* has no hard upper limit on
sequence length — the FlashAttention kernel is O(N) memory, so a
70k-cell slice that would OOM under the naive attention path runs
cleanly here. Whatever ``max_seq_len`` cap LUNA's
``LinearAttentionTransformer`` carries (70000) does NOT apply to
this module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAMultiheadAttention(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention`` that calls
    ``F.scaled_dot_product_attention`` explicitly.

    Args:
        embed_dim: per-token output dim (== q_dim).
        num_heads: number of attention heads. ``embed_dim % num_heads``
            must be zero.
        dropout: attention dropout probability. Applied inside SDPA
            only during ``self.training``.
        bias: include bias on q/k/v/out projections (default True, to
            match nn.MultiheadAttention's default).
        kdim, vdim: separate K/V input widths. Defaults to ``embed_dim``
            (the self-attention case). When given, the K and V proj
            layers accept the wider input — same as MHA's ``kdim``/
            ``vdim`` args. Used by Perceiver's cross-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads "
                f"{num_heads} (per-head dim must be an integer)."
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.kdim = int(kdim) if kdim is not None else self.embed_dim
        self.vdim = int(vdim) if vdim is not None else self.embed_dim
        self.dropout = float(dropout)

        # Explicit q/k/v projections (one per stream). We DON'T fuse
        # them into a single packed qkv linear because the cross-
        # attention case (kdim != embed_dim) has different K/V input
        # widths from Q.
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, N_q, embed_dim).
            key:   (B, N_kv, kdim).
            value: (B, N_kv, vdim).
            key_padding_mask: (B, N_kv) bool, True at PAD positions
                (matches nn.MultiheadAttention's convention).
            need_weights: must be False — exposing weights forces
                materialising the N×N matrix and we'd silently leave
                the memory-efficient path. Kept in the signature so
                MHA call sites work unchanged, but a True value
                raises a clear error.
            attn_mask: additive attention bias of shape (N_q, N_kv)
                or (B*num_heads, N_q, N_kv). Float; -inf at
                disallowed positions. Combined with key_padding_mask
                via addition when both are supplied.

        Returns:
            (out, None) — out has shape (B, N_q, embed_dim).
        """
        if need_weights:
            raise RuntimeError(
                "SDPAMultiheadAttention does not return attention "
                "weights — exposing them forces materialising the "
                "N×N matrix and defeats the FlashAttention memory-"
                "efficient path. Use a dedicated weights-extracting "
                "module if you need that signal."
            )

        B, N_q, _ = query.shape
        N_kv = key.shape[1]
        H, D = self.num_heads, self.head_dim

        # Project + split heads. Each becomes (B, H, N, D).
        q = self.q_proj(query).view(B, N_q, H, D).transpose(1, 2)
        k = self.k_proj(key  ).view(B, N_kv, H, D).transpose(1, 2)
        v = self.v_proj(value).view(B, N_kv, H, D).transpose(1, 2)

        # Build the SDPA attn_mask. SDPA accepts:
        #   * bool — True at ALLOWED positions (note: OPPOSITE of MHA's
        #     key_padding_mask convention).
        #   * float — additive bias; -inf at disallowed positions.
        # When BOTH key_padding_mask and attn_mask are given, we have
        # to merge them. Use float form for the merge so both signals
        # combine correctly.
        sdpa_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None or attn_mask is not None:
            mask_dtype = q.dtype
            # Start with a zero bias of the right broadcast shape.
            bias_shape = (B, 1, 1, N_kv)
            sdpa_mask = torch.zeros(
                bias_shape, dtype=mask_dtype, device=q.device,
            )
            if key_padding_mask is not None:
                if key_padding_mask.dtype != torch.bool:
                    raise TypeError(
                        f"key_padding_mask must be bool (True at PAD); "
                        f"got dtype {key_padding_mask.dtype}."
                    )
                # True at PAD → add -inf (disallow attending to PAD).
                # finfo.min instead of -inf to avoid NaNs in fp16 paths
                # if a row attends *only* to PAD positions; that case is
                # zero rows in practice, but the SDPA fp16 fast path
                # has been observed to NaN with -inf.
                neg_inf = torch.finfo(mask_dtype).min
                pad_bias = torch.where(
                    key_padding_mask,
                    torch.tensor(neg_inf, dtype=mask_dtype, device=q.device),
                    torch.tensor(0.0,     dtype=mask_dtype, device=q.device),
                )
                # (B, N_kv) → (B, 1, 1, N_kv) broadcast across heads + queries.
                sdpa_mask = sdpa_mask + pad_bias.view(B, 1, 1, N_kv)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    neg_inf = torch.finfo(mask_dtype).min
                    attn_bias = torch.where(
                        attn_mask,
                        torch.tensor(0.0,     dtype=mask_dtype, device=q.device),
                        torch.tensor(neg_inf, dtype=mask_dtype, device=q.device),
                    )
                else:
                    attn_bias = attn_mask.to(mask_dtype)
                # Broadcast to (B, H, N_q, N_kv). attn_mask can be
                # (N_q, N_kv) or (B*H, N_q, N_kv) — handle both.
                if attn_bias.dim() == 2:
                    attn_bias = attn_bias.view(1, 1, N_q, N_kv)
                elif attn_bias.dim() == 3:
                    attn_bias = attn_bias.view(B, H, N_q, N_kv)
                else:
                    raise ValueError(
                        f"attn_mask must have rank 2 or 3; got rank "
                        f"{attn_bias.dim()}."
                    )
                sdpa_mask = sdpa_mask + attn_bias

        # SDPA call. is_causal=False because cell-cell attention has
        # no temporal/causal structure (cells are a set).
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        # (B, H, N_q, D) → (B, N_q, H*D = embed_dim).
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        out = self.out_proj(out)
        return out, None
