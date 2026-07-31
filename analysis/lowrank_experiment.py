"""Low-rank interaction model (factorization-machine style) for Dota drafts.

The pairwise model has ~16,000 free interaction weights, one per hero pair, each
estimated from a few dozen matches. This instead gives every hero a small
embedding and *derives* every pair's interaction as a dot product, so the
parameter count drops from 16,000 to ~127 x 3d while still expressing synergy
and counter effects for all pairs.

Structure (antisymmetric by construction, exactly like the linear model):

    logit = b + sum_R w_i - sum_D w_j
              + [ S(R) - S(D) ]                  synergy
              + [ P_R . Q_D - P_D . Q_R ]        counter

    S(T) = sum_{i<j in T} <v_i, v_j> = 0.5 * ( ||sum_T v||^2 - sum_T ||v||^2 )

Swapping the two teams negates every term except b, so the mirrored-draft
identity still holds exactly.

Trained with Adam and manual gradients (no torch in this environment).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features import build_hero_index, load_matches, parse_team  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402

RNG = np.random.default_rng(0)


def build_arrays(df, hero_index):
    R = np.array([[hero_index[h] for h in parse_team(t)] for t in df["radiant_team"]])
    D = np.array([[hero_index[h] for h in parse_team(t)] for t in df["dire_team"]])
    y = df["radiant_win"].to_numpy().astype(np.float64)
    return R, D, y


def forward(params, R, D):
    w, v, p, q, b = params
    lin = w[R].sum(1) - w[D].sum(1)

    vR, vD = v[R], v[D]                       # (n, 5, d)
    svR, svD = vR.sum(1), vD.sum(1)           # (n, d)
    synR = 0.5 * ((svR ** 2).sum(1) - (vR ** 2).sum((1, 2)))
    synD = 0.5 * ((svD ** 2).sum(1) - (vD ** 2).sum((1, 2)))

    PR, PD = p[R].sum(1), p[D].sum(1)
    QR, QD = q[R].sum(1), q[D].sum(1)
    ctr = (PR * QD).sum(1) - (PD * QR).sum(1)

    return lin + synR - synD + ctr + b, (svR, svD, vR, vD, PR, PD, QR, QD)


def fit(R, D, y, n_heroes, d=6, l2=3e-3, lr=0.02, epochs=60, batch=512, val=0.1, verbose=False):
    n_val = int(len(y) * val)
    Rtr, Dtr, ytr = R[:-n_val], D[:-n_val], y[:-n_val]
    Rva, Dva, yva = R[-n_val:], D[-n_val:], y[-n_val:]

    scale = 0.01
    w = np.zeros(n_heroes)
    v = RNG.normal(0, scale, (n_heroes, d))
    p = RNG.normal(0, scale, (n_heroes, d))
    q = RNG.normal(0, scale, (n_heroes, d))
    b = float(np.log(ytr.mean() / (1 - ytr.mean())))
    params = [w, v, p, q, b]

    m = [np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0 for x in params]
    vv = [np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0 for x in params]
    t = 0
    best = (np.inf, None)

    for epoch in range(epochs):
        order = RNG.permutation(len(ytr))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            Rb, Db, yb = Rtr[idx], Dtr[idx], ytr[idx]
            logit, cache = forward(params, Rb, Db)
            svR, svD, vR, vD, PR, PD, QR, QD = cache
            pr = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
            g = (pr - yb) / len(yb)                       # (nb,)

            gw = np.zeros_like(w); gv = np.zeros_like(v)
            gp = np.zeros_like(p); gq = np.zeros_like(q)

            np.add.at(gw, Rb, g[:, None])
            np.add.at(gw, Db, -g[:, None])

            # d(syn)/dv_i = +/- (sum_T v - v_i)
            np.add.at(gv, Rb, g[:, None, None] * (svR[:, None, :] - vR))
            np.add.at(gv, Db, -g[:, None, None] * (svD[:, None, :] - vD))

            # d(ctr) terms
            np.add.at(gp, Rb, g[:, None, None] * QD[:, None, :])
            np.add.at(gq, Db, g[:, None, None] * PR[:, None, :])
            np.add.at(gp, Db, -g[:, None, None] * QR[:, None, :])
            np.add.at(gq, Rb, -g[:, None, None] * PD[:, None, :])

            gb = g.sum()

            gw += l2 * w; gv += l2 * v; gp += l2 * p; gq += l2 * q
            grads = [gw, gv, gp, gq, gb]

            t += 1
            for i, (par, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                vv[i] = 0.999 * vv[i] + 0.001 * (gr * gr)
                mh = m[i] / (1 - 0.9 ** t)
                vh = vv[i] / (1 - 0.999 ** t)
                params[i] = par - lr * mh / (np.sqrt(vh) + 1e-8)
            w, v, p, q, b = params

        logit, _ = forward(params, Rva, Dva)
        pv = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        ll = log_loss(yva, np.clip(pv, 1e-6, 1 - 1e-6))
        if ll < best[0]:
            best = (ll, [x.copy() if isinstance(x, np.ndarray) else x for x in params])
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d} val logloss {ll:.5f}")

    return best[1]


def main() -> None:
    df = load_matches("data/matches.csv.gz", game_mode=22)
    heroes = set()
    for r, dd in zip(df["radiant_team"], df["dire_team"]):
        heroes.update(parse_team(r)); heroes.update(parse_team(dd))
    hi = build_hero_index(heroes)
    R, D, y = build_arrays(df, hi)

    split = int(len(y) * 0.8)
    Rtr, Dtr, ytr = R[:split], D[:split], y[:split]
    Rte, Dte, yte = R[split:], D[split:], y[split:]
    print(f"{len(ytr):,} train / {len(yte):,} test, {len(hi)} heroes\n")

    for d in (2, 4, 8):
        params = fit(Rtr, Dtr, ytr, len(hi), d=d)
        logit, _ = forward(params, Rte, Dte)
        pt = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        print(
            f"  d={d}:  acc={accuracy_score(yte, pt >= .5):.4f}  "
            f"logloss={log_loss(yte, pt):.4f}  auc={roc_auc_score(yte, pt):.4f}"
        )

    # Antisymmetry check on the largest model
    logit_fwd, _ = forward(params, Rte[:200], Dte[:200])
    logit_rev, _ = forward(params, Dte[:200], Rte[:200])
    b = params[4]
    print(f"\n  antisymmetry max error: {np.abs((logit_fwd - b) + (logit_rev - b)).max():.2e}")


if __name__ == "__main__":
    main()
