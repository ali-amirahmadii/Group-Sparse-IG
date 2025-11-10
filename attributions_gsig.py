{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww51000\viewh28800\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # attributions_gsig.py\
# ------------------------------------------------------------\
# Integrated Gradients with multiple baselines + GS-IG\
# (manifold-aware baseline and group-sparse path optimization)\
#\
# Requirements: torch (Optuna optional; falls back to grid search)\
# ------------------------------------------------------------\
\
from __future__ import annotations\
\
import math\
import warnings\
from dataclasses import dataclass\
from enum import Enum, auto\
from typing import Callable, Iterable, Optional, Tuple, Dict\
\
import numpy as np\
import torch\
from torch import nn\
from torch.autograd import grad\
\
\
# ------------------------------- #\
# Utilities & typing\
# ------------------------------- #\
\
Tensor = torch.Tensor\
\
\
def choose_device(pref: Optional[str] = None) -> torch.device:\
    """Pick a device; defaults to CUDA if available."""\
    if pref is not None:\
        return torch.device(pref)\
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")\
\
\
def set_deterministic(seed: int = 42) -> None:\
    torch.manual_seed(seed)\
    np.random.seed(seed)\
    try:\
        import random\
        random.seed(seed)\
    except Exception:\
        pass\
    torch.backends.cudnn.deterministic = True\
    torch.backends.cudnn.benchmark = False\
\
\
@dataclass\
class SpecialTokenIDs:\
    pad_id: int\
    mask_id: Optional[int] = None\
    cls_id: Optional[int] = None\
    sep_id: Optional[int] = None\
    unk_id: Optional[int] = None\
\
\
class SeqBaseline(Enum):\
    ZERO = auto()\
    PAD_ALL = auto()\
    MASK_ALL = auto()\
    PAD_PER_TRAJECTORY = auto()\
    MANIFOLD_MEAN_EMBED = auto()  # (position-wise mean embedding from validation)\
\
\
class TabBaseline(Enum):\
    ZERO = auto()\
    EMPIRICAL_MEAN = auto()\
\
\
@dataclass\
class IGConfig:\
    steps: int = 50                    # Riemann samples along the path\
    theta: float = 1.0                 # path schedule exponent; 1.0 = straight line\
    optimized_theta: bool = False      # enable per-sample path selection\
    theta_min: float = 0.1\
    theta_max: float = 5.0\
    theta_trials: int = 10             # if optimized_theta, number of trials\
    lambda_group: float = 1.0          # group-l2 weight for GS objective\
    lambda_l1: float = 0.0             # optional elementwise L1\
    exclude_token_ids_from_groups: Tuple[int, ...] = tuple()  # e.g., (pad_id, cls_id, sep_id)\
    use_optuna_if_available: bool = True\
\
\
# ------------------------------- #\
# Manifold-aware embedding baseline\
# ------------------------------- #\
\
@torch.no_grad()\
def compute_positionwise_mean_embedding(\
    model: nn.Module,\
    embedding_module: nn.Module,\
    val_loader: Iterable,\
    ids_index: int,\
    device: torch.device,\
) -> Tensor:\
    """\
    Compute a position-wise mean of the embedding-module *output* on validation data.\
    Assumes fixed sequence length across batches (pad to a constant length).\
\
    Returns:\
        mean_embed: (1, L, d) float tensor on `device`\
    """\
    model.eval()\
    embs = []\
    for batch in val_loader:\
        ids = batch[ids_index].to(device)          # (B, L)\
        out = embedding_module(ids)                # (B, L, d)\
        embs.append(out.detach())\
    stacked = torch.cat(embs, dim=0)               # (N, L, d)\
    mean_embed = stacked.mean(dim=0, keepdim=True) # (1, L, d)\
    return mean_embed\
\
\
@torch.no_grad()\
def compute_tabular_empirical_mean(\
    val_loader: Iterable,\
    tab_index: int,\
    device: torch.device\
) -> Tensor:\
    """\
    Compute an empirical mean vector for tabular inputs from validation set.\
    Returns:\
        mean_tab: (1, D) float tensor on `device`\
    """\
    xs = []\
    for batch in val_loader:\
        x_tab = batch[tab_index]\
        xs.append(x_tab)\
    X = torch.cat(xs, dim=0).to(torch.float32)  # (N, D)\
    mean_tab = X.mean(dim=0, keepdim=True).to(device)\
    return mean_tab\
\
\
# ------------------------------- #\
# Baseline generators\
# ------------------------------- #\
\
def generate_seq_baseline(\
    ids: Tensor,                      # (B, L) int\
    embedding_module: nn.Module,      # module whose forward(ids) -> (B, L, d)\
    baseline_type: SeqBaseline,\
    specials: SpecialTokenIDs,\
    manifold_mean_embed: Optional[Tensor] = None  # (1, L, d)\
) -> Tensor:\
    """\
    Generate a sequence baseline in *embedding space* (output of embedding_module).\
    """\
    B, L = ids.shape\
    if baseline_type == SeqBaseline.ZERO:\
        return torch.zeros_like(embedding_module(ids))\
    if baseline_type == SeqBaseline.PAD_ALL:\
        pad = torch.full_like(ids, specials.pad_id)\
        return embedding_module(pad)\
    if baseline_type == SeqBaseline.MASK_ALL:\
        if specials.mask_id is None:\
            raise ValueError("MASK_ALL requested but mask_id=None in SpecialTokenIDs.")\
        mask = torch.full_like(ids, specials.mask_id)\
        return embedding_module(mask)\
    if baseline_type == SeqBaseline.PAD_PER_TRAJECTORY:\
        # Replace all non-special tokens with [PAD], preserve the sequence length.\
        pad_equivalent = ids.clone()\
        for b in range(B):\
            for t in range(L):\
                tok = pad_equivalent[b, t].item()\
                if tok not in \{specials.pad_id, specials.cls_id, specials.sep_id\}:\
                    pad_equivalent[b, t] = specials.pad_id\
        return embedding_module(pad_equivalent)\
    if baseline_type == SeqBaseline.MANIFOLD_MEAN_EMBED:\
        if manifold_mean_embed is None:\
            raise ValueError("MANIFOLD_MEAN_EMBED requested but manifold_mean_embed=None.")\
        # Broadcast to batch size\
        return manifold_mean_embed.expand(B, -1, -1).to(ids.device)\
    raise NotImplementedError(baseline_type)\
\
\
def generate_tab_baseline(\
    x_tab: Tensor,                      # (B, D)\
    baseline_type: TabBaseline,\
    empirical_mean_tab: Optional[Tensor] = None  # (1, D)\
) -> Tensor:\
    if baseline_type == TabBaseline.ZERO:\
        return torch.zeros_like(x_tab)\
    if baseline_type == TabBaseline.EMPIRICAL_MEAN:\
        if empirical_mean_tab is None:\
            raise ValueError("EMPIRICAL_MEAN requested but empirical_mean_tab=None.")\
        return empirical_mean_tab.expand(x_tab.size(0), -1).to(x_tab.device)\
    raise NotImplementedError(baseline_type)\
\
\
# ------------------------------- #\
# Path schedule & hooking\
# ------------------------------- #\
\
def alpha_schedule(steps: int, theta: float, device: torch.device) -> Tensor:\
    """\
    Return alpha[0..K] with alpha_0=0, alpha_K=1 for re-parameterized schedule t^theta.\
    """\
    t = torch.linspace(0.0, 1.0, steps + 1, device=device)\
    return t ** float(theta)\
\
\
def make_replace_hook(replacement: Tensor) -> Callable:\
    """\
    Forward hook to replace an embedding module's output with `replacement`.\
    """\
    def _hook(module, inputs, output):  # pylint: disable=unused-argument\
        return replacement\
    return _hook\
\
\
# ------------------------------- #\
# IG core\
# ------------------------------- #\
\
def _scalar_score_from_model(\
    model: nn.Module,\
    forward_fn: Callable[..., Tuple[Tensor, ...]],\
    ids: Tensor,\
    attn_mask: Optional[Tensor],\
    pad_aux: Optional[Tensor],\
    x_tab: Optional[Tensor],\
) -> Tensor:\
    """\
    Calls forward_fn(model, ids, attn_mask, pad_aux, x_tab) and returns a scalar.\
    By default expects the first returned tensor to be a probability/logit with shape (B, 1) or (B,).\
    """\
    out = forward_fn(model, ids, attn_mask, pad_aux, x_tab)\
    if isinstance(out, Tensor):\
        y = out\
    else:\
        # use the first element\
        y = out[0]\
    y = y.squeeze()\
    if y.ndim == 0:\
        return y\
    # If it's (B,) or (B,1): take sum to get a scalar (batch is typically 1 for IG)\
    return y.sum()\
\
\
def integrated_gradients(\
    *,\
    model: nn.Module,\
    embedding_module: nn.Module,\
    forward_fn: Callable[..., Tuple[Tensor, ...]],\
    ids: Tensor,                        # (B, L) int\
    attn_mask: Optional[Tensor],        # e.g. (B, 1, L, L) or anything your model expects\
    pad_aux: Optional[Tensor],          # any extra arg your model needs (pass-through)\
    x_tab: Optional[Tensor],            # (B, D) or None\
    seq_baseline: Tensor,               # (B, L, d) baseline in embedding space\
    tab_baseline: Optional[Tensor],     # (B, D) or None\
    steps: int = 50,\
    theta: float = 1.0,\
    device: Optional[torch.device] = None\
) -> Tuple[Tensor, Optional[Tensor]]:\
    """\
    Compute IG for embeddings (and optionally tabular) along schedule alpha(t)=t^theta.\
    Returns:\
        ig_embed: (B, L, d)\
        ig_tab:   (B, D) or None\
    """\
    device = device or ids.device\
    model.eval()\
\
    # Prepare endpoints\
    with torch.no_grad():\
        E = embedding_module(ids).detach()       # (B, L, d) original point\
        E0 = seq_baseline.detach()               # (B, L, d)\
\
        if x_tab is not None and tab_baseline is not None:\
            X = x_tab.detach()\
            X0 = tab_baseline.detach()\
        else:\
            X = None\
            X0 = None\
\
    alphas = alpha_schedule(steps, theta, device)\
    grads_embed = []\
    grads_tab = []\
\
    # Cache ref for faster hook registration\
    emb_layer = embedding_module\
\
    # Disable CuDNN RNN just for attribution runs (optional)\
    with torch.backends.cudnn.flags(enabled=False):\
        for k in range(1, steps + 1):\
            a = alphas[k]\
            da = a - alphas[k - 1]\
\
            # Build path points\
            E_k = (E0 + a * (E - E0)).clone().detach().requires_grad_(True)\
            if X is not None:\
                X_k = (X0 + a * (X - X0)).clone().detach().requires_grad_(True)\
            else:\
                X_k = None\
\
            # Override the embedding output for this forward pass\
            handle = emb_layer.register_forward_hook(make_replace_hook(E_k))\
\
            model.zero_grad(set_to_none=True)\
            score = _scalar_score_from_model(model, forward_fn, ids, attn_mask, pad_aux, X_k)\
            g_embed = grad(score, E_k, retain_graph=False, create_graph=False)[0]\
            grads_embed.append(g_embed * da)\
\
            if X_k is not None:\
                g_tab = grad(score, X_k, retain_graph=False, create_graph=False)[0]\
                grads_tab.append(g_tab * da)\
\
            # clean the hook\
            handle.remove()\
\
    total_grad_embed = torch.stack(grads_embed, dim=0).sum(dim=0)  # (B, L, d)\
    ig_embed = (E - E0) * total_grad_embed\
\
    if X is not None and X0 is not None:\
        total_grad_tab = torch.stack(grads_tab, dim=0).sum(dim=0)  # (B, D)\
        ig_tab = (X - X0) * total_grad_tab\
    else:\
        ig_tab = None\
\
    return ig_embed, ig_tab\
\
\
# ------------------------------- #\
# GS-PO-IG (theta selection with group sparsity)\
# ------------------------------- #\
\
def token_group_l2(ig_embed: Tensor, valid_token_mask: Optional[Tensor] = None) -> Tensor:\
    """\
    Group L2 over embedding dims per token. Returns sum over tokens (batch-aware).\
    ig_embed: (B, L, d)\
    valid_token_mask: (B, L) bool, True means include; if None includes all.\
    """\
    B, L, d = ig_embed.shape\
    token_norms = torch.norm(ig_embed, p=2, dim=-1)  # (B, L)\
    if valid_token_mask is not None:\
        token_norms = token_norms * valid_token_mask.float()\
    # sum over tokens then batch\
    return token_norms.sum(dim=-1).mean()  # scalar (batch-mean)\
\
\
def l1_norm(ig_embed: Tensor, valid_token_mask: Optional[Tensor] = None) -> Tensor:\
    val = ig_embed.abs()\
    if valid_token_mask is not None:\
        val = val * valid_token_mask[..., None].float()\
    return val.sum(dim=[1, 2]).mean()\
\
\
def select_theta_gs(\
    *,\
    model: nn.Module,\
    embedding_module: nn.Module,\
    forward_fn: Callable[..., Tuple[Tensor, ...]],\
    ids: Tensor,\
    attn_mask: Optional[Tensor],\
    pad_aux: Optional[Tensor],\
    x_tab: Optional[Tensor],\
    seq_baseline: Tensor,\
    tab_baseline: Optional[Tensor],\
    cfg: IGConfig,\
    device: torch.device\
) -> float:\
    """\
    Select theta per input by minimizing group-sparse surrogate:\
      L(theta) = lambda_group * sum_j ||IG[j,:]||_2 + lambda_l1 * ||IG||_1\
    """\
    candidate_thetas: Iterable[float]\
    use_optuna = cfg.use_optuna_if_available\
    if cfg.optimized_theta and use_optuna:\
        try:\
            import optuna  # type: ignore\
\
            def objective(trial):\
                theta = trial.suggest_float("theta", cfg.theta_min, cfg.theta_max)\
                ig_emb, _ = integrated_gradients(\
                    model=model,\
                    embedding_module=embedding_module,\
                    forward_fn=forward_fn,\
                    ids=ids,\
                    attn_mask=attn_mask,\
                    pad_aux=pad_aux,\
                    x_tab=x_tab,\
                    seq_baseline=seq_baseline,\
                    tab_baseline=tab_baseline,\
                    steps=cfg.steps,\
                    theta=theta,\
                    device=device,\
                )\
                # valid tokens: exclude specials if provided\
                valid_mask = None\
                if cfg.exclude_token_ids_from_groups:\
                    with torch.no_grad():\
                        tok_ids = ids.detach()\
                        exclude = torch.zeros_like(tok_ids, dtype=torch.bool)\
                        for sid in cfg.exclude_token_ids_from_groups:\
                            if sid is not None:\
                                exclude |= (tok_ids == sid)\
                        valid_mask = ~exclude\
                gl = token_group_l2(ig_emb, valid_mask)\
                l1 = l1_norm(ig_emb, valid_mask)\
                loss = cfg.lambda_group * gl + cfg.lambda_l1 * l1\
                return float(loss.detach().cpu().item())\
\
            sampler = optuna.samplers.TPESampler(seed=42)\
            study = optuna.create_study(direction="minimize", sampler=sampler)\
            study.optimize(objective, n_trials=cfg.theta_trials)\
            return float(study.best_params["theta"])\
        except Exception:\
            warnings.warn("Optuna not available; falling back to grid search.")\
            use_optuna = False\
\
    # Grid search fallback or when optimized_theta=True but optuna disabled.\
    if cfg.optimized_theta and not use_optuna:\
        # Log grid in [theta_min, theta_max]\
        ts = np.linspace(math.log(cfg.theta_min), math.log(cfg.theta_max), cfg.theta_trials)\
        candidate_thetas = [float(math.exp(v)) for v in ts]\
    else:\
        # No optimization requested\
        candidate_thetas = [cfg.theta]\
\
    best_theta = None\
    best_loss = float("inf")\
\
    for theta in candidate_thetas:\
        ig_emb, _ = integrated_gradients(\
            model=model,\
            embedding_module=embedding_module,\
            forward_fn=forward_fn,\
            ids=ids,\
            attn_mask=attn_mask,\
            pad_aux=pad_aux,\
            x_tab=x_tab,\
            seq_baseline=seq_baseline,\
            tab_baseline=tab_baseline,\
            steps=cfg.steps,\
            theta=theta,\
            device=device,\
        )\
        valid_mask = None\
        if cfg.exclude_token_ids_from_groups:\
            with torch.no_grad():\
                tok_ids = ids.detach()\
                exclude = torch.zeros_like(tok_ids, dtype=torch.bool)\
                for sid in cfg.exclude_token_ids_from_groups:\
                    if sid is not None:\
                        exclude |= (tok_ids == sid)\
                valid_mask = ~exclude\
\
        gl = token_group_l2(ig_emb, valid_mask)\
        l1 = l1_norm(ig_emb, valid_mask)\
        loss = cfg.lambda_group * gl + cfg.lambda_l1 * l1\
\
        loss_v = float(loss.detach().cpu().item())\
        if loss_v < best_loss:\
            best_loss = loss_v\
            best_theta = theta\
\
    assert best_theta is not None\
    return float(best_theta)\
\
\
# ------------------------------- #\
# Public API: compute IG/GS-PO-IG\
# ------------------------------- #\
\
def compute_ig_for_sample(\
    *,\
    model: nn.Module,\
    embedding_module: nn.Module,\
    forward_fn: Callable[..., Tuple[Tensor, ...]],\
    ids: Tensor,                         # (1, L) int\
    attn_mask: Optional[Tensor],         # whatever your model expects\
    pad_aux: Optional[Tensor],\
    x_tab: Optional[Tensor],             # (1, D) or None\
    seq_baseline_type: SeqBaseline,\
    tab_baseline_type: Optional[TabBaseline] = None,\
    manifold_mean_embed: Optional[Tensor] = None,   # (1, L, d)\
    empirical_mean_tab: Optional[Tensor] = None,    # (1, D)\
    cfg: Optional[IGConfig] = None,\
    specials: Optional[SpecialTokenIDs] = None,\
    device: Optional[torch.device] = None\
) -> Dict[str, Tensor]:\
    """\
    Compute IG (and optionally GS-PO-IG) for a single sample.\
\
    Returns a dict with:\
        'ig_embed': (1, L, d),\
        'ig_tab': (1, D) or None,\
        'theta': float used,\
        'token_scores_l2': (1, L) group L2 per token (for readability)\
    """\
    cfg = cfg or IGConfig()\
    device = device or ids.device\
    ids = ids.to(device)\
    attn_mask = None if attn_mask is None else attn_mask.to(device)\
    pad_aux = None if pad_aux is None else pad_aux.to(device)\
    x_tab = None if x_tab is None else x_tab.to(device)\
\
    # Prepare baselines\
    seq_baseline = generate_seq_baseline(\
        ids=ids,\
        embedding_module=embedding_module,\
        baseline_type=seq_baseline_type,\
        specials=(specials or SpecialTokenIDs(pad_id=0)),\
        manifold_mean_embed=manifold_mean_embed,\
    )\
    if tab_baseline_type is not None and x_tab is not None:\
        tab_baseline = generate_tab_baseline(\
            x_tab=x_tab, baseline_type=tab_baseline_type, empirical_mean_tab=empirical_mean_tab\
        )\
    else:\
        tab_baseline = None\
\
    # Exclude specials from group sparsity if provided\
    exclude_ids = tuple(\
        sid for sid in (specials.pad_id if specials else None,\
                        specials.cls_id if specials else None,\
                        specials.sep_id if specials else None)\
        if sid is not None\
    )\
    cfg.exclude_token_ids_from_groups = exclude_ids\
\
    # Theta selection (optional GS-PO-IG)\
    theta = cfg.theta\
    if cfg.optimized_theta:\
        theta = select_theta_gs(\
            model=model,\
            embedding_module=embedding_module,\
            forward_fn=forward_fn,\
            ids=ids,\
            attn_mask=attn_mask,\
            pad_aux=pad_aux,\
            x_tab=x_tab,\
            seq_baseline=seq_baseline,\
            tab_baseline=tab_baseline,\
            cfg=cfg,\
            device=device,\
        )\
\
    # Final IG with chosen theta\
    ig_embed, ig_tab = integrated_gradients(\
        model=model,\
        embedding_module=embedding_module,\
        forward_fn=forward_fn,\
        ids=ids,\
        attn_mask=attn_mask,\
        pad_aux=pad_aux,\
        x_tab=x_tab,\
        seq_baseline=seq_baseline,\
        tab_baseline=tab_baseline,\
        steps=cfg.steps,\
        theta=theta,\
        device=device,\
    )\
\
    # Token-level L2 scores (readability; not complete)\
    token_scores = torch.norm(ig_embed, p=2, dim=-1)  # (1, L)\
\
    return \{\
        "ig_embed": ig_embed.detach(),\
        "ig_tab": None if ig_tab is None else ig_tab.detach(),\
        "theta": torch.tensor([theta], device=device),\
        "token_scores_l2": token_scores.detach(),\
    \}\
\
\
# ------------------------------- #\
# Example forward_fn adapter\
# ------------------------------- #\
#\
# def forward_fn_example(model, ids, attn_mask, pad_aux, x_tab):\
#     """\
#     Example for a binary classifier returning probability (B,1).\
#     Convert to logit for more stable gradients.\
#     """\
#     prob, *_ = model(ids, attn_mask, pad_aux, x_tab)  # -> (B,1)\
#     prob = prob.clamp(1e-6, 1 - 1e-6)\
#     logit = torch.log(prob / (1 - prob))\
#     return logit  # IG code will sum across batch to get a scalar\
#\
# ------------------------------- #\
\
\
# ------------------------------- #\
# Minimal usage sketch (commented)\
# ------------------------------- #\
#\
# if __name__ == "__main__":\
#     set_deterministic(42)\
#     device = choose_device()\
#\
#     # model: your trained nn.Module\
#     # embedding_module: the module producing (B,L,d) from ids (e.g., model.bert.embedding)\
#     # val_loader: yields batches with (ids, ..., x_tab)\
#     # test_batch: a single batch to attribute\
#\
#     specials = SpecialTokenIDs(pad_id=1, mask_id=2, cls_id=0, sep_id=3)\
#\
#     # Compute manifold-aware means (optional but recommended)\
#     mean_embed = compute_positionwise_mean_embedding(\
#         model=model,\
#         embedding_module=embedding_module,\
#         val_loader=val_loader,\
#         ids_index=0,\
#         device=device,\
#     )\
#     mean_tab = compute_tabular_empirical_mean(val_loader, tab_index=4, device=device)\
#\
#     # Config for GS-IG\
#     cfg = IGConfig(\
#         steps=50,\
#         optimized_theta=True,\
#         theta_min=0.1,\
#         theta_max=5.0,\
#         theta_trials=10,\
#         lambda_group=1.0,\
#         lambda_l1=0.0,\
#         use_optuna_if_available=True,\
#     )\
#\
#     # One sample (batch size 1 recommended)\
#     ids, attnmask, _, pad_aux, x_tab = next(iter(test_loader))\
#     result = compute_ig_for_sample(\
#         model=model,\
#         embedding_module=embedding_module,\
#         forward_fn=forward_fn_example,\
#         ids=ids.to(device),\
#         attn_mask=attnmask.to(device),\
#         pad_aux=pad_aux.to(device),\
#         x_tab=x_tab.to(device),\
#         seq_baseline_type=SeqBaseline.MANIFOLD_MEAN_EMBED,\
#         tab_baseline_type=TabBaseline.EMPIRICAL_MEAN,\
#         manifold_mean_embed=mean_embed,\
#         empirical_mean_tab=mean_tab,\
#         cfg=cfg,\
#         specials=specials,\
#         device=device,\
#     )\
#\
#     ig_embed = result["ig_embed"]\
#     ig_tab = result["ig_tab"]\
#     theta_used = float(result["theta"].item())\
#     token_scores = result["token_scores_l2"]\
#\
#     print("Theta used:", theta_used)\
#     print("IG embed shape:", ig_embed.shape)\
#     if ig_tab is not None:\
#         print("IG tab shape:", ig_tab.shape)}