import torch
from typing import Optional, Tuple, Dict
from utils.SNPA import SNPA
from utils.NNLS import NNLS


def _pair_indices(r: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (I, J) for all pairs 0<=i<j<r, as int64 tensors.
    """
    idx = torch.triu_indices(r, r, offset=1, device=device)
    return idx[0], idx[1]


def _P(W: torch.Tensor, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    P(W) = [w_i ∘ w_j] for all i<j.
    W: (m, r)
    returns: (m, r*(r-1)/2)
    """
    return W[:, I] * W[:, J]


def _S(H: torch.Tensor, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    S(H) = [h_i ∘ h_j] for all i<j.
    H: (r, n)
    returns: (r*(r-1)/2, n)
    """
    return H[I, :] * H[J, :]


@torch.no_grad()
def BiNMF(
    X: torch.Tensor,
    r: int,
    iters: int = 200,
    lam: float = 1.0,          # ASC augmentation parameter (lambda in the tex)
    gamma: float = 1.0,        # gamma in the tex
    W: Optional[torch.Tensor] = None,
    H: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Lambda: Optional[torch.Tensor] = None,
    mu0: float = 1.0,          # initial step length; mu_t decreases to 0
    project_lambda_nonneg: bool = False,
    clamp_Z_to_gamma_SH: bool = False,  # optional "min(gamma*S(H), ...)" projection note in tex
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    BiNMF using HALS updates as in main.tex:

        X ≈ W H + P(W) Z
        with constraint Z <= gamma S(H) handled via Lagrange multipliers Lambda >= 0.

    Shapes (without ASC augmentation):
        X: (m, n)
        W: (m, r)
        H: (r, n)
        Z: (rt, n) where rt = r*(r-1)//2
        Lambda: (rt, n)

    If lam>0, we do ASC augmentation:
        Xbar = [X; lam*1^T], enforce last row of Wbar to lam*1^T (like your current code).
    """
    device = X.device
    dtype = X.dtype
    m0, n = X.shape

    # --- ASC augmentation (optional), consistent with your current implementation style
    if lam and lam > 0:
        en = torch.ones((1, n), device=device, dtype=dtype)
        X_work = torch.cat([X, lam * en], dim=0)     # (m0+1, n)
        m = m0 + 1
    else:
        X_work = X
        m = m0

    rt = r * (r - 1) // 2
    I, J = _pair_indices(r, device=device)

    # --- init
    g = torch.Generator(device=device)
    g.manual_seed(0)

    if W is None:
        # W = torch.rand((m, r), device=device, dtype=dtype, generator=g).clamp_min_(0)

        options = {}
        options.setdefault('dtype', dtype)
        K, H = SNPA(X, r, options)
        W0 = X[:, K]

        optionsNNLS = {'dtype': dtype}
        H, ـ, ـ = NNLS(W0, X, optionsNNLS)

        er = torch.ones(1, r).type(W0.type())
        W = torch.cat((W0, lam * er), dim=0)
    else:
        W = W.to(device=device, dtype=dtype).clone()

    if H is None:
        H = torch.rand((r, n), device=device, dtype=dtype, generator=g).clamp_min_(0)
    else:
        H = H.to(device=device, dtype=dtype).clone()

    if Z is None:
        Omega = _P(W, I, J)
        optionsNNLS = {'dtype': dtype}
        Z, _, _ = NNLS(Omega, X_work - W @ H, optionsNNLS)
    else:
        Z = Z.to(device=device, dtype=dtype).clone()

    if Lambda is None:
        Lambda = torch.zeros((rt, n), device=device, dtype=dtype)
    else:
        Lambda = Lambda.to(device=device, dtype=dtype).clone()

    # Enforce ASC row on W if augmented
    if lam and lam > 0:
        W[-1, :] = lam

    # --- logs
    rel_err = torch.zeros((iters,), device=device, dtype=dtype)

    for t in range(iters):
        # step length mu_t -> 0

        mu_t = 1 / (t + 1)
        mu_t = 1 / ((t + 1) * (t + 1) ** 0.5)
        # mu_t = 1 / (t + 1)**2

        # Current bilinear basis Omega = P(W)
        Omega = _P(W, I, J)  # (m, rt)

        # ======================
        # Update W (column HALS)
        # ======================
        # We'll use current full approximation, then build the per-l residual pieces.
        WH = W @ H                 # (m, n)
        OZ = Omega @ Z             # (m, n)
        approx = WH + OZ

        for l in range(r):
            # Indices q != l
            others = [q for q in range(r) if q != l]
            W_oth = W[:, others]           # (m, r-1)
            H_oth = H[others, :]           # (r-1, n)

            # Pair rows (l,q) in Z/Lambda for q != l
            # Build list of pair-row indices in the same order as `others`.
            pair_rows = []
            for q in others:
                if l < q:
                    # row corresponds to (l,q)
                    pair_rows.append(((I == l) & (J == q)).nonzero(as_tuple=False).item())
                else:
                    # row corresponds to (q,l)
                    pair_rows.append(((I == q) & (J == l)).nonzero(as_tuple=False).item())
            pair_rows = torch.tensor(pair_rows, device=device, dtype=torch.long)

            Z_l = Z[pair_rows, :]          # (r-1, n) rows are z^(l,q)
            # Residual that removes everything not depending on w_l:
            # R_l = X - (approx - w_l h_l - sum_q (w_l∘w_q) z^(l,q))
            w_l = W[:, l]                  # (m,)
            h_l = H[l, :]                  # (n,)
            bil_l = (w_l[:, None] * (W_oth)) @ torch.zeros((len(others), n), device=device, dtype=dtype)  # unused
            # compute sum_q (w_l∘w_q) z_lq = (w_l[:,None] * W_oth) @ Z_l
            sum_bilin_l = (w_l[:, None] * W_oth) @ Z_l  # (m, n)
            R_l = X_work - (approx - (w_l[:, None] @ h_l[None, :]) - sum_bilin_l)

            # Numerator: R_l h_l^T + sum_q (R_l z_lq^T) ∘ w_q
            num0 = R_l @ h_l               # (m,)
            A = R_l @ Z_l.T                # (m, r-1)
            num1 = (A * W_oth).sum(dim=1)  # (m,)
            numer = num0 + num1

            # Denominator:
            den0 = (h_l @ h_l).clamp_min(eps)  # scalar
            # 2 * sum_q w_q * (z_lq h_l^T)
            s = (Z_l @ h_l)                    # (r-1,)
            den1 = 2.0 * (W_oth * s[None, :]).sum(dim=1)  # (m,)

            # sum_{q,p != l} (w_q ∘ w_p) (z_lq z_lp^T)
            # Let G = Z_l Z_l^T, then per-row i: w_i^T G w_i
            G = Z_l @ Z_l.T                    # (r-1, r-1)
            den2 = ((W_oth @ G) * W_oth).sum(dim=1)        # (m,)

            denom = (den0 + den1 + den2).clamp_min(eps)
            W[:, l] = torch.clamp(numer / denom, min=0.0)

            # Re-enforce ASC row for W if augmented
            if lam and lam > 0:
                W[-1, l] = lam

            # Update cached approx for next l (simple but safe):
            Omega = _P(W, I, J)
            WH = W @ H
            OZ = Omega @ Z
            approx = WH + OZ

        # ======================
        # Update H (row HALS)
        # ======================
        Omega = _P(W, I, J)
        OZ = Omega @ Z

        for l in range(r):
            others = [q for q in range(r) if q != l]
            W_oth = W[:, others]   # (m, r-1)
            H_oth = H[others, :]   # (r-1, n)

            # Pair rows (l,q) in Lambda for q != l, aligned with `others`
            pair_rows = []
            for q in others:
                if l < q:
                    pair_rows.append(((I == l) & (J == q)).nonzero(as_tuple=False).item())
                else:
                    pair_rows.append(((I == q) & (J == l)).nonzero(as_tuple=False).item())
            pair_rows = torch.tensor(pair_rows, device=device, dtype=torch.long)

            Lam_l = Lambda[pair_rows, :]  # (r-1, n)
            w_l = W[:, l]
            h_l = H[l, :]

            # B_l = X - P(W)Z - sum_{k != l} w_k h_k
            #     = X - OZ - (W_oth H_oth)
            B_l = X_work - OZ - (W_oth @ H_oth)

            # numerator: w_l^T B_l + gamma * sum_{p != l} lambda^(l,p) ∘ h^(p)
            num0 = w_l @ B_l                      # (n,)
            num1 = gamma * (Lam_l * H_oth).sum(dim=0)  # (n,)
            numer = num0 + num1

            denom = (w_l @ w_l).clamp_min(eps)    # scalar
            H[l, :] = torch.clamp(numer / denom, min=0.0)

        # ======================
        # Update Z (row HALS)
        # ======================
        Omega = _P(W, I, J)
        resid = X_work - (W @ H) - (Omega @ Z)   # (m, n)

        for k in range(rt):
            omega_k = Omega[:, k]                # (m,)
            denom = (omega_k @ omega_k).clamp_min(eps)
            # Coordinate HALS step:
            numer = omega_k @ resid - Lambda[k, :]     # (n,)
            delta = numer / denom
            z_new = torch.clamp(Z[k, :] + delta, min=0.0)

            if clamp_Z_to_gamma_SH:
                # Optional projection mentioned in the tex note:
                # z^(l,p) <- min(gamma*h^l∘h^p, z_update)
                z_cap = gamma * (H[I[k], :] * H[J[k], :])
                z_new = torch.minimum(z_new, z_cap)

            # Update residual efficiently: resid = X - WH - OmegaZ
            # If Z[k] changes by (z_new - z_old), OmegaZ changes by omega_k*(z_new - z_old)
            z_old = Z[k, :]
            resid += omega_k[:, None] * (z_old - z_new)[None, :]
            Z[k, :] = z_new

        # ======================
        # Update Lambda
        # ======================
        SH = _S(H, I, J)  # (rt, n)
        Lambda = Lambda + mu_t * (Z - gamma * SH)
        if project_lambda_nonneg:
            Lambda = torch.clamp(Lambda, min=0.0)

        # ======================
        # Logging
        # ======================
        Omega = _P(W, I, J)
        rel_err[t] = torch.linalg.norm(X_work - (W @ H) - (Omega @ Z)) / torch.linalg.norm(X_work).clamp_min(eps)

    info = {"rel_err": rel_err, "Lambda": Lambda}
    return W, H, Z, Omega, rel_err

