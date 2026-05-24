"""SE(2)-equivariant vector-neuron transformer backbone — scGG
fundamental method #3.

Status: SCAFFOLD. The config flag (``backbone: "vn_transformer"``)
is wired through ``diffusion_model.py`` but the actual architecture
is not yet implemented; instantiating this module raises
NotImplementedError with a clear message.

Why the scaffold-first pattern
------------------------------
A correct vector-neuron transformer is ~500-800 LOC of careful
equivariance-preserving primitives (VN-Linear, VN-ReLU, VN-LayerNorm,
VN-Attention). Each primitive must be unit-tested for equivariance
(rotate the input, run the layer, compare to applying the rotation
to the output). Getting this wrong silently breaks the whole
inductive-bias argument the architecture exists to make.

The scaffold separates two concerns:
  1. Config surface — covered now. ``cfg.model.backbone`` accepts
     "vn_transformer" without crashing the config loader; selecting
     it fails fast with a clear message at construction time.
  2. Architecture — left for a focused implementation session where
     the equivariance tests are written first and the layers built
     against them.

Planned architecture (Deng et al. 2021 "Vector Neurons" + standard
attention)
---------------------------------------------------------------------
Per cell we maintain (s, V) where s ∈ R^S is a SCALAR feature vector
and V ∈ R^{C × 2} is a stack of C 2D VECTOR channels. All operations
preserve the contract: rotating the INPUT V by R ∈ SO(2) rotates the
OUTPUT V by the same R, while s is unchanged.

Primitives:
  - VN-Linear(c→c'): W ∈ R^{c'×c}, applied as V' = W V along the
    channel axis. Equivariance: (W V) R = W (V R) since R acts on
    the right of V.
  - VN-Nonlinearity: project each vector channel onto a learned
    half-space (defined by a learned direction u ∈ R^2, computed
    from the SCALAR features so u is a scalar function of equivariant
    inputs → invariant → fine to use as a gate).
  - VN-Inner-product (V·V): scalar (B, N, C, 2) · (B, N, C, 2) sums
    to scalar — invariant. Used to inject scalars from vector
    features.
  - VN-Attention: queries / keys are formed from (s, V); the
    attention weights are SCALAR (invariant); the values are V
    (equivariant); the weighted sum stays equivariant.

Output:
  - Positions are read off the FIRST vector channel of the final
    layer's V (one channel of (B, N, 2)).
  - Node features are read off the SCALAR stream s.

References
----------
- Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A.,
  Guibas, L. J. (2021). "Vector Neurons: A General Framework for SO(3)-
  Equivariant Networks." ICCV. arXiv:2104.12229.
- Same construction restricted to SO(2) is what we use here.
"""

from __future__ import annotations

import torch.nn as nn


class VNTransformerBackbone(nn.Module):
    """Placeholder for the SE(2)-equivariant vector-neuron transformer
    backbone. Raises on construction so selecting this backbone via
    config produces a clear actionable error rather than a silent
    misbehaviour.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "model.backbone='vn_transformer' is not yet implemented. "
            "The config surface is in place (see "
            "configs/model/default.yaml::vn_transformer) but the "
            "actual vector-neuron attention layers haven't been built. "
            "This is scGG fundamental method #3 — schedule a focused "
            "session to implement VN-Linear / VN-ReLU / VN-Attention "
            "primitives + equivariance tests, then this stub. For now "
            "use backbone='luna_transformer' or 'egnn'."
        )
