"""
attributions_gsig.py
------------------------------------------------------------
Integrated Gradients with multiple baselines + GS-IG
(manifold-aware baseline and group-sparse path optimization)

This module implements:
  - Position-wise manifold-aware embedding baselines
  - Empirical mean baselines for tabular inputs
  - Standard Integrated Gradients (IG) in embedding space
  - Group-sparse path-optimized IG (GS-IG) via theta selection

Requirements:
  - torch
  - numpy
  - (optional) optuna   # for theta optimization; falls back to grid search

The core functions:

  - compute_positionwise_mean_embedding(...)
  - compute_tabular_empirical_mean(...)
  - compute_ig_for_sample(...)

See the bottom of the file for a minimal usage example.
------------------------------------------------------------
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable, Optional, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Utilities & typing helpers
# ---------------------------------------------------------------------------


def choose_device(pref: Optional[str] = None) -> torch.device:
    """
    Pick a device; defaults to CUDA if available.

    Args:
        pref: Optional string like "cuda", "cpu", or "cuda:1".
              If None, uses "cuda" when available, else "cpu".

    Returns:
        A torch.device.
    """
    if pref is not None:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic(seed: int = 42) -> None:
    """
    Set seeds for reproducibility (PyTorch, NumPy, Python's random).

    Note:
        This does not make everything perfectly reproducible across GPUs,
        but is a reasonable baseline.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SpecialTokenIDs:
    """
    Container for special token IDs used in sequence baselines.

    Attributes:
        pad_id: ID of the [PAD] token (required).
        mask_id: ID of the [MASK] token, if available.
        cls_id: ID of the [CLS] token, if available.
        sep_id: ID of the [SEP] token, if available.
        unk_id: ID of the [UNK] token, if available.
    """

    pad_id: int
    mask_id: Optional[int] = None
    cls_id: Optional[int] = None
    sep_id: Optional[int] = None
    unk_id: Optional[int] = None


class SeqBaseline(Enum):
    """
    Different strategies for sequence baselines in embedding space.

    All baselines are defined in *embedding space*, i.e., after the
    token IDs have been passed through `embedding_module`.
    """

    ZERO = auto()  # All-zero embedding baseline
    PAD_ALL = auto()  # All tokens set to [PAD] and embedded
    MASK_ALL = auto()  # All tokens set to [MASK] and embedded
    PAD_PER_TRAJECTORY = auto()  # All non-special tokens replaced with [PAD]
    MANIFOLD_MEAN_EMBED = (
        auto()
    )  # Position-wise mean embedding from validation set


class TabBaseline(Enum):
    """
    Baselines for tabular (continuous) inputs.
    """

    ZERO = auto()  # All-zero tabular inputs
    EMPIRICAL_MEAN = auto()  # Empirical mean over validation set


@dataclass
class IGConfig:
    """
    Configuration for Integrated Gradients and GS-IG.

    Attributes:
        steps: Number of Riemann steps along the path.
        theta: Path schedule exponent; 1.0 corresponds to a straight line.
        optimized_theta: If True, selects theta per sample using group-sparse loss.
        theta_min: Lower bound for theta search.
        theta_max: Upper bound for theta search.
        theta_trials: Number of theta candidates (or Optuna trials).
        lambda_group: Weight for group-L2 token-level sparsity.
        lambda_l1: Optional weight for elementwise L1.
        exclude_token_ids_from_groups: Token IDs to exclude from the group sparsity
                                       (e.g., PAD, CLS, SEP).
        use_optuna_if_available: Try to use Optuna if installed; otherwise use grid.
    """

    steps: int = 50
    theta: float = 1.0
    optimized_theta: bool = False
    theta_min: float = 0.1
    theta_max: float = 5.0
    theta_trials: int = 10
    lambda_group: float = 1.0
    lambda_l1: float = 0.0
    exclude_token_ids_from_groups: Tuple[int, ...] = tuple()
    use_optuna_if_available: bool = True


# ---------------------------------------------------------------------------
# Manifold-aware embedding and tabular baselines
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_positionwise_mean_embedding(
    model: nn.Module,
    embedding_module: nn.Module,
    val_loader: Iterable,
    ids_index: int,
    device: torch.device,
) -> Tensor:
    """
    Compute a position-wise mean of the embedding-module *output* on validation data.

    This is the manifold-aware baseline used in the paper: we average the embeddings
    at each sequence position over the validation set.

    Args:
        model: The trained model (only used in eval mode; parameters are untouched).
        embedding_module: Module whose forward(ids) -> (B, L, d).
        val_loader: Iterable yielding batches; batch[ids_index] must be token IDs (B, L).
        ids_index: Index in each batch tuple/list where the token IDs live.
        device: Device to move inputs/outputs to.

    Returns:
        mean_embed: Tensor of shape (1, L, d) on `device`.
    """
    model.eval()
    embs = []

    for batch in val_loader:
        ids = batch[ids_index].to(device)  # (B, L)
        out = embedding_module(ids)  # (B, L, d)
        embs.append(out.detach())

    stacked = torch.cat(embs, dim=0)  # (N, L, d)
    mean_embed = stacked.mean(dim=0, keepdim=True)  # (1, L, d)
    return mean_embed


@torch.no_grad()
def compute_tabular_empirical_mean(
    val_loader: Iterable,
    tab_index: int,
    device: torch.device,
) -> Tensor:
    """
    Compute an empirical mean vector for tabular inputs from the validation set.

    Args:
        val_loader: Iterable yielding batches; batch[tab_index] must be (B, D).
        tab_index: Index in each batch tuple/list where the tabular features live.
        device: Device for the returned tensor.

    Returns:
        mean_tab: Tensor of shape (1, D) on `device`.
    """
    xs = []
    for batch in val_loader:
        x_tab = batch[tab_index]
        xs.append(x_tab)

    X = torch.cat(xs, dim=0).to(torch.float32)  # (N, D)
    mean_tab = X.mean(dim=0, keepdim=True).to(device)
    return mean_tab


# ---------------------------------------------------------------------------
# Baseline generators
# ---------------------------------------------------------------------------


def generate_seq_baseline(
    ids: Tensor,  # (B, L) int
    embedding_module: nn.Module,  # module whose forward(ids) -> (B, L, d)
    baseline_type: SeqBaseline,
    specials: SpecialTokenIDs,
    manifold_mean_embed: Optional[Tensor] = None,  # (1, L, d)
) -> Tensor:
    """
    Generate a sequence baseline in *embedding space* (output of embedding_module).

    Args:
        ids: Token IDs of shape (B, L).
        embedding_module: Module producing embeddings (B, L, d) from IDs.
        baseline_type: One of SeqBaseline.
        specials: SpecialTokenIDs configuration.
        manifold_mean_embed: For MANIFOLD_MEAN_EMBED, a tensor of shape (1, L, d).

    Returns:
        Tensor of shape (B, L, d) representing baseline embeddings.
    """
    B, L = ids.shape

    if baseline_type == SeqBaseline.ZERO:
        return torch.zeros_like(embedding_module(ids))

    if baseline_type == SeqBaseline.PAD_ALL:
        pad = torch.full_like(ids, specials.pad_id)
        return embedding_module(pad)

    if baseline_type == SeqBaseline.MASK_ALL:
        if specials.mask_id is None:
            raise ValueError("MASK_ALL requested but mask_id=None in SpecialTokenIDs.")
        mask = torch.full_like(ids, specials.mask_id)
        return embedding_module(mask)

    if baseline_type == SeqBaseline.PAD_PER_TRAJECTORY:
        # Replace all non-[PAD]/[CLS]/[SEP] tokens with [PAD], preserving length.
        pad_equivalent = ids.clone()
        exclude = torch.zeros_like(pad_equivalent, dtype=torch.bool)

        # Tokens we keep as-is:
        keep_ids = [specials.pad_id]
        if specials.cls_id is not None:
            keep_ids.append(specials.cls_id)
        if specials.sep_id is not None:
            keep_ids.append(specials.sep_id)

        for sid in keep_ids:
            exclude |= pad_equivalent.eq(sid)

        # Everything not in keep_ids becomes PAD
        pad_equivalent = torch.where(exclude, pad_equivalent, torch.full_like(pad_equivalent, specials.pad_id))
        return embedding_module(pad_equivalent)

    if baseline_type == SeqBaseline.MANIFOLD_MEAN_EMBED:
        if manifold_mean_embed is None:
            raise ValueError("MANIFOLD_MEAN_EMBED requested but manifold_mean_embed=None.")
        # Broadcast to batch size
        return manifold_mean_embed.expand(B, -1, -1).to(ids.device)

    raise NotImplementedError(baseline_type)


def generate_tab_baseline(
    x_tab: Tensor,  # (B, D)
    baseline_type: TabBaseline,
    empirical_mean_tab: Optional[Tensor] = None,  # (1, D)
) -> Tensor:
    """
    Generate a baseline for tabular inputs.

    Args:
        x_tab: Tabular input of shape (B, D).
        baseline_type: One of TabBaseline.
        empirical_mean_tab: (1, D) empirical mean vector for EMPIRICAL_MEAN.

    Returns:
        Tensor of shape (B, D) representing baseline tabular input.
    """
    if baseline_type == TabBaseline.ZERO:
        return torch.zeros_like(x_tab)

    if baseline_type == TabBaseline.EMPIRICAL_MEAN:
        if empirical_mean_tab is None:
            raise ValueError("EMPIRICAL_MEAN requested but empirical_mean_tab=None.")
        return empirical_mean_tab.expand(x_tab.size(0), -1).to(x_tab.device)

    raise NotImplementedError(baseline_type)


# ---------------------------------------------------------------------------
# Path schedule & embedding hook
# ---------------------------------------------------------------------------


def alpha_schedule(steps: int, theta: float, device: torch.device) -> Tensor:
    """
    Compute alpha[0..K] with alpha_0=0, alpha_K=1 for schedule alpha(t)=t^theta.

    Args:
        steps: Number of Riemann steps.
        theta: Path exponent; 1.0 = straight line.
        device: Device for returned tensor.

    Returns:
        1D Tensor of shape (steps+1,) on `device`.
    """
    t = torch.linspace(0.0, 1.0, steps + 1, device=device)
    return t**float(theta)


def make_replace_hook(replacement: Tensor) -> Callable:
    """
    Create a forward hook that replaces a module's output with `replacement`.

    Used to override the embedding module's output along the IG path.
    """

    def _hook(module, inputs, output):  # pylint: disable=unused-argument
        return replacement

    return _hook


# ---------------------------------------------------------------------------
# IG core
# ---------------------------------------------------------------------------


def _scalar_score_from_model(
    model: nn.Module,
    forward_fn: Callable[..., Tuple[Tensor, ...]],
    ids: Tensor,
    attn_mask: Optional[Tensor],
    pad_aux: Optional[Tensor],
    x_tab: Optional[Tensor],
) -> Tensor:
    """
    Calls forward_fn(model, ids, attn_mask, pad_aux, x_tab) and returns a scalar.

    Convention:
        - forward_fn should return either:
            * a single Tensor (B,) or (B,1), or
            * a tuple where the first element is such a tensor.
        - We sum over batch to get a scalar (IG is typically run with B=1).

    Returns:
        Scalar Tensor.
    """
    out = forward_fn(model, ids, attn_mask, pad_aux, x_tab)

    if isinstance(out, Tensor):
        y = out
    else:
        # Use the first element
        y = out[0]

    y = y.squeeze()

    if y.ndim == 0:
        return y

    # If it's (B,) or (B,1): take sum to get a scalar
    return y.sum()


def integrated_gradients(
    *,
    model: nn.Module,
    embedding_module: nn.Module,
    forward_fn: Callable[..., Tuple[Tensor, ...]],
    ids: Tensor,  # (B, L) int
    attn_mask: Optional[Tensor],
    pad_aux: Optional[Tensor],
    x_tab: Optional[Tensor],  # (B, D) or None
    seq_baseline: Tensor,  # (B, L, d) baseline in embedding space
    tab_baseline: Optional[Tensor],  # (B, D) or None
    steps: int = 50,
    theta: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute Integrated Gradients for embeddings (and optionally tabular data)
    along schedule alpha(t)=t^theta.

    Args:
        model: Trained nn.Module.
        embedding_module: Module producing (B, L, d) from ids.
        forward_fn: Function adapting model forward to return the scalar target.
        ids: Token IDs of shape (B, L).
        attn_mask: Attention mask as expected by the model (or None).
        pad_aux: Any extra tensor argument your model needs (or None).
        x_tab: Tabular input (B, D) or None.
        seq_baseline: Baseline embedding (B, L, d).
        tab_baseline: Baseline tabular input (B, D) or None.
        steps: Number of Riemann steps.
        theta: Path exponent.
        device: Device on which to run computations.

    Returns:
        ig_embed: Tensor of shape (B, L, d).
        ig_tab: Tensor of shape (B, D) or None.
    """
    device = device or ids.device
    model.eval()

    # Prepare endpoints
    with torch.no_grad():
        E = embedding_module(ids).detach()  # (B, L, d) original point
        E0 = seq_baseline.detach()  # (B, L, d)

        if x_tab is not None and tab_baseline is not None:
            X = x_tab.detach()
            X0 = tab_baseline.detach()
        else:
            X = None
            X0 = None

    alphas = alpha_schedule(steps, theta, device)
    grads_embed = []
    grads_tab = []

    emb_layer = embedding_module

    # Disable CuDNN (optional) to avoid subtle non-determinism in some models
    with torch.backends.cudnn.flags(enabled=False):
        for k in range(1, steps + 1):
            a = alphas[k]
            da = a - alphas[k - 1]

            # Build path points
            E_k = (E0 + a * (E - E0)).clone().detach().requires_grad_(True)
            if X is not None:
                X_k = (X0 + a * (X - X0)).clone().detach().requires_grad_(True)
            else:
                X_k = None

            # Override the embedding output for this forward pass
            handle = emb_layer.register_forward_hook(make_replace_hook(E_k))

            model.zero_grad(set_to_none=True)
            score = _scalar_score_from_model(model, forward_fn, ids, attn_mask, pad_aux, X_k)

            g_embed = grad(score, E_k, retain_graph=False, create_graph=False)[0]
            grads_embed.append(g_embed * da)

            if X_k is not None:
                g_tab = grad(score, X_k, retain_graph=False, create_graph=False)[0]
                grads_tab.append(g_tab * da)

            # Clean hook
            handle.remove()

    total_grad_embed = torch.stack(grads_embed, dim=0).sum(dim=0)  # (B, L, d)
    ig_embed = (E - E0) * total_grad_embed

    if X is not None and X0 is not None:
        total_grad_tab = torch.stack(grads_tab, dim=0).sum(dim=0)  # (B, D)
        ig_tab = (X - X0) * total_grad_tab
    else:
        ig_tab = None

    return ig_embed, ig_tab


# ---------------------------------------------------------------------------
# GS-IG: group-sparse path optimization in theta
# ---------------------------------------------------------------------------


def token_group_l2(ig_embed: Tensor, valid_token_mask: Optional[Tensor] = None) -> Tensor:
    """
    Group L2 over embedding dims per token, then sum over tokens and average over batch.

    Args:
        ig_embed: Tensor of shape (B, L, d).
        valid_token_mask: Bool tensor (B, L); True=include token. If None, includes all.

    Returns:
        Scalar Tensor (batch-mean of sum over token norms).
    """
    token_norms = torch.norm(ig_embed, p=2, dim=-1)  # (B, L)
    if valid_token_mask is not None:
        token_norms = token_norms * valid_token_mask.float()
    return token_norms.sum(dim=-1).mean()


def l1_norm(ig_embed: Tensor, valid_token_mask: Optional[Tensor] = None) -> Tensor:
    """
    Elementwise L1 norm over IG attributions (optionally restricted to valid tokens).
    """
    val = ig_embed.abs()
    if valid_token_mask is not None:
        val = val * valid_token_mask[..., None].float()
    return val.sum(dim=[1, 2]).mean()


def select_theta_gs(
    *,
    model: nn.Module,
    embedding_module: nn.Module,
    forward_fn: Callable[..., Tuple[Tensor, ...]],
    ids: Tensor,
    attn_mask: Optional[Tensor],
    pad_aux: Optional[Tensor],
    x_tab: Optional[Tensor],
    seq_baseline: Tensor,
    tab_baseline: Optional[Tensor],
    cfg: IGConfig,
    device: torch.device,
) -> float:
    """
    Select theta per input by minimizing a group-sparse surrogate:

        L(theta) = lambda_group * sum_j ||IG[j,:]||_2 + lambda_l1 * ||IG||_1

    The function uses Optuna if available (and cfg.use_optuna_if_available=True),
    otherwise falls back to a log-spaced grid search over [theta_min, theta_max].
    """
    candidate_thetas: Iterable[float]
    use_optuna = cfg.use_optuna_if_available

    if cfg.optimized_theta and use_optuna:
        try:
            import optuna  # type: ignore

            def objective(trial):
                theta = trial.suggest_float("theta", cfg.theta_min, cfg.theta_max)
                ig_emb, _ = integrated_gradients(
                    model=model,
                    embedding_module=embedding_module,
                    forward_fn=forward_fn,
                    ids=ids,
                    attn_mask=attn_mask,
                    pad_aux=pad_aux,
                    x_tab=x_tab,
                    seq_baseline=seq_baseline,
                    tab_baseline=tab_baseline,
                    steps=cfg.steps,
                    theta=theta,
                    device=device,
                )

                valid_mask = None
                if cfg.exclude_token_ids_from_groups:
                    with torch.no_grad():
                        tok_ids = ids.detach()
                        exclude = torch.zeros_like(tok_ids, dtype=torch.bool)
                        for sid in cfg.exclude_token_ids_from_groups:
                            exclude |= tok_ids.eq(sid)
                        valid_mask = ~exclude

                gl = token_group_l2(ig_emb, valid_mask)
                l1 = l1_norm(ig_emb, valid_mask)
                loss = cfg.lambda_group * gl + cfg.lambda_l1 * l1
                return float(loss.detach().cpu().item())

            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=cfg.theta_trials)
            return float(study.best_params["theta"])

        except Exception:
            warnings.warn("Optuna not available; falling back to grid search.")
            use_optuna = False

    # Grid search fallback or when optimized_theta=True but optuna disabled.
    if cfg.optimized_theta and not use_optuna:
        # Log grid in [theta_min, theta_max]
        ts = np.linspace(math.log(cfg.theta_min), math.log(cfg.theta_max), cfg.theta_trials)
        candidate_thetas = [float(math.exp(v)) for v in ts]
    else:
        # No optimization requested
        candidate_thetas = [cfg.theta]

    best_theta = None
    best_loss = float("inf")

    for theta in candidate_thetas:
        ig_emb, _ = integrated_gradients(
            model=model,
            embedding_module=embedding_module,
            forward_fn=forward_fn,
            ids=ids,
            attn_mask=attn_mask,
            pad_aux=pad_aux,
            x_tab=x_tab,
            seq_baseline=seq_baseline,
            tab_baseline=tab_baseline,
            steps=cfg.steps,
            theta=theta,
            device=device,
        )

        valid_mask = None
        if cfg.exclude_token_ids_from_groups:
            with torch.no_grad():
                tok_ids = ids.detach()
                exclude = torch.zeros_like(tok_ids, dtype=torch.bool)
                for sid in cfg.exclude_token_ids_from_groups:
                    exclude |= tok_ids.eq(sid)
                valid_mask = ~exclude

        gl = token_group_l2(ig_emb, valid_mask)
        l1 = l1_norm(ig_emb, valid_mask)
        loss = cfg.lambda_group * gl + cfg.lambda_l1 * l1

        loss_v = float(loss.detach().cpu().item())
        if loss_v < best_loss:
            best_loss = loss_v
            best_theta = theta

    assert best_theta is not None
    return float(best_theta)


# ---------------------------------------------------------------------------
# Public API: compute IG / GS-IG for a single sample
# ---------------------------------------------------------------------------


def compute_ig_for_sample(
    *,
    model: nn.Module,
    embedding_module: nn.Module,
    forward_fn: Callable[..., Tuple[Tensor, ...]],
    ids: Tensor,  # (1, L) int recommended
    attn_mask: Optional[Tensor],
    pad_aux: Optional[Tensor],
    x_tab: Optional[Tensor],  # (1, D) or None
    seq_baseline_type: SeqBaseline,
    tab_baseline_type: Optional[TabBaseline] = None,
    manifold_mean_embed: Optional[Tensor] = None,  # (1, L, d)
    empirical_mean_tab: Optional[Tensor] = None,  # (1, D)
    cfg: Optional[IGConfig] = None,
    specials: Optional[SpecialTokenIDs] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """
    Compute IG (and optionally GS-IG) for a single sample.

    This is the main high-level function intended for external use.

    Args:
        model: Trained nn.Module.
        embedding_module: Module mapping token IDs to embeddings (B, L, d).
        forward_fn: Adapter function that defines the scalar output to explain.
                    Signature: forward_fn(model, ids, attn_mask, pad_aux, x_tab)
        ids: Token IDs (1, L) is recommended for per-patient attributions.
        attn_mask: Whatever attention mask your model expects (or None).
        pad_aux: Any extra tensor your model needs (or None).
        x_tab: Tabular input (1, D) or None.
        seq_baseline_type: Sequence baseline choice (see SeqBaseline).
        tab_baseline_type: Tabular baseline choice (see TabBaseline), or None.
        manifold_mean_embed: Position-wise mean embedding (1, L, d) for MANIFOLD_MEAN_EMBED.
        empirical_mean_tab: Mean tabular vector (1, D) for EMPIRICAL_MEAN baseline.
        cfg: IGConfig; if None, uses defaults.
        specials: SpecialTokenIDs; required for certain baseline types.
        device: Device for computation; if None, inferred from ids.device.

    Returns:
        A dictionary with:
            "ig_embed": (1, L, d) embedding attributions,
            "ig_tab":   (1, D) tabular attributions or None,
            "theta":    tensor([theta]) used,
            "token_scores_l2": (1, L) token-level L2 scores (for readability).
    """
    cfg = cfg or IGConfig()
    device = device or ids.device

    ids = ids.to(device)
    attn_mask = None if attn_mask is None else attn_mask.to(device)
    pad_aux = None if pad_aux is None else pad_aux.to(device)
    x_tab = None if x_tab is None else x_tab.to(device)

    specials = specials or SpecialTokenIDs(pad_id=0)

    # Prepare sequence baseline
    seq_baseline = generate_seq_baseline(
        ids=ids,
        embedding_module=embedding_module,
        baseline_type=seq_baseline_type,
        specials=specials,
        manifold_mean_embed=manifold_mean_embed,
    )

    # Prepare tabular baseline (if requested)
    if tab_baseline_type is not None and x_tab is not None:
        tab_baseline = generate_tab_baseline(
            x_tab=x_tab,
            baseline_type=tab_baseline_type,
            empirical_mean_tab=empirical_mean_tab,
        )
    else:
        tab_baseline = None

    # Exclude special tokens from group sparsity
    exclude_ids = tuple(
        sid
        for sid in (
            specials.pad_id,
            specials.cls_id,
            specials.sep_id,
        )
        if sid is not None
    )
    cfg.exclude_token_ids_from_groups = exclude_ids

    # Theta selection (optional GS-IG)
    theta = cfg.theta
    if cfg.optimized_theta:
        theta = select_theta_gs(
            model=model,
            embedding_module=embedding_module,
            forward_fn=forward_fn,
            ids=ids,
            attn_mask=attn_mask,
            pad_aux=pad_aux,
            x_tab=x_tab,
            seq_baseline=seq_baseline,
            tab_baseline=tab_baseline,
            cfg=cfg,
            device=device,
        )

    # Final IG with chosen theta
    ig_embed, ig_tab = integrated_gradients(
        model=model,
        embedding_module=embedding_module,
        forward_fn=forward_fn,
        ids=ids,
        attn_mask=attn_mask,
        pad_aux=pad_aux,
        x_tab=x_tab,
        seq_baseline=seq_baseline,
        tab_baseline=tab_baseline,
        steps=cfg.steps,
        theta=theta,
        device=device,
    )

    # Token-level L2 scores (for readability/debugging)
    token_scores = torch.norm(ig_embed, p=2, dim=-1)  # (1, L)

    return {
        "ig_embed": ig_embed.detach(),
        "ig_tab": None if ig_tab is None else ig_tab.detach(),
        "theta": torch.tensor([theta], device=device),
        "token_scores_l2": token_scores.detach(),
    }


# ---------------------------------------------------------------------------
# Example forward_fn adapter & minimal usage sketch
# ---------------------------------------------------------------------------

# def forward_fn_example(model, ids, attn_mask, pad_aux, x_tab):
#     """
#     Example forward_fn for a binary classifier returning probability (B, 1).
#     We convert probability to logits for more stable gradients.
#     """
#     prob, *_ = model(ids, attn_mask, pad_aux, x_tab)  # -> (B, 1)
#     prob = prob.clamp(1e-6, 1 - 1e-6)
#     logit = torch.log(prob / (1 - prob))
#     return logit  # IG code will sum across batch to get a scalar


# if __name__ == "__main__":
#     # This is a minimal sketch; replace placeholders with your actual objects.
#     set_deterministic(42)
#     device = choose_device()
#
#     # model: your trained nn.Module
#     # embedding_module: the module producing (B, L, d) from ids
#     # val_loader: yields batches with (ids, ..., x_tab)
#     # test_loader: yields batches you want to explain
#
#     specials = SpecialTokenIDs(pad_id=1, mask_id=2, cls_id=0, sep_id=3)
#
#     # Compute manifold-aware means (recommended)
#     mean_embed = compute_positionwise_mean_embedding(
#         model=model,
#         embedding_module=embedding_module,
#         val_loader=val_loader,
#         ids_index=0,
#         device=device,
#     )
#
#     mean_tab = compute_tabular_empirical_mean(
#         val_loader=val_loader,
#         tab_index=4,  # adjust index to your batch structure
#         device=device,
#     )
#
#     # Config for GS-IG (set optimized_theta=False for vanilla IG)
#     cfg = IGConfig(
#         steps=50,
#         optimized_theta=True,
#         theta_min=0.1,
#         theta_max=5.0,
#         theta_trials=10,
#         lambda_group=1.0,
#         lambda_l1=0.0,
#         use_optuna_if_available=True,
#     )
#
#     # One sample (batch size 1 recommended)
#     ids, attnmask, _, pad_aux, x_tab = next(iter(test_loader))
#
#     result = compute_ig_for_sample(
#         model=model,
#         embedding_module=embedding_module,
#         forward_fn=forward_fn_example,
#         ids=ids.to(device),
#         attn_mask=attnmask.to(device),
#         pad_aux=pad_aux.to(device),
#         x_tab=x_tab.to(device),
#         seq_baseline_type=SeqBaseline.MANIFOLD_MEAN_EMBED,
#         tab_baseline_type=TabBaseline.EMPIRICAL_MEAN,
#         manifold_mean_embed=mean_embed,
#         empirical_mean_tab=mean_tab,
#         cfg=cfg,
#         specials=specials,
#         device=device,
#     )
#
#     ig_embed = result["ig_embed"]
#     ig_tab = result["ig_tab"]
#     theta_used = float(result["theta"].item())
#     token_scores = result["token_scores_l2"]
#
#     print("Theta used:", theta_used)
#     print("IG embed shape:", ig_embed.shape)
#     if ig_tab is not None:
#         print("IG tab shape:", ig_tab.shape)
