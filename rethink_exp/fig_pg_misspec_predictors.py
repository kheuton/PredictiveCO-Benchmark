"""
Plot the Phase 2 winner predictor for every method on pg_misspec.

Each method's winner uses a `dense, n_layers=1` predictor: nn.Linear(2, 1)
on features [1, x]. So the prediction is an affine function of x:

    yhat(x) = (w0 + b) + w1 * x

We overlay it on the true piecewise-linear cost (kink at x=0.5), and the
data scatter (train+val). The decision is z=1 iff yhat<=0 (MINIMIZE),
so the "learned threshold" is x* where yhat(x*)=0.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
P2_JSON = REPO / "bench_p2_best_val.json"
OUT_DIR = REPO / "results"
OUT_DIR.mkdir(exist_ok=True)


# ---- True data-generating process (matches PGMisspec defaults) ----
def true_cost(x, m0=-4.0, m=0.0, c0=-0.2, t_kink=0.55):
    return np.where(x < t_kink, m0 * x + 2.0, m * (x - t_kink) + c0)


def gen_data(n_train=200, n_val=200, rand_seed=2023, sd=0.5, alpha=1.0):
    """Reproduce the train+val portion of PGMisspec for plotting."""
    rng = np.random.RandomState(rand_seed)
    np.random.seed(rand_seed)
    N = n_train + n_val
    x = np.random.uniform(0.0, 2.0, N).astype(np.float32)
    c_clean = true_cost(x).astype(np.float32)
    zeta = np.random.exponential(sd, N).astype(np.float32) - sd
    gamma = np.random.normal(0.0, sd, N).astype(np.float32)
    eps = np.sqrt(alpha) * zeta + np.sqrt(1.0 - alpha) * gamma
    Y = c_clean + eps
    # Wrapper assigns the first n_val to val, rest to train.
    return x[n_val:], Y[n_val:], x[:n_val], Y[:n_val]


# ---- Load predictor weights ----
def load_winner_predictor(run_dir):
    """Return (slope, intercept) of the affine pred yhat(x) = intercept + slope*x."""
    ckpt_path = REPO / run_dir / "checkpoints" / "tr_pred_best.pt"
    if not ckpt_path.exists():
        return None
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # net.0 is Linear(2, 1): yhat = w0 * 1 + w1 * x + b
    w = sd["net.0.weight"].flatten().tolist()  # [w0, w1]
    b = sd["net.0.bias"].flatten().tolist()[0]
    intercept = w[0] + b
    slope = w[1]
    return slope, intercept


def main():
    with open(P2_JSON) as f:
        best = json.load(f)

    methods = [
        "mse", "identity", "dfl",
        "spo", "blackbox", "nce",
        "pointLTR", "pairLTR", "listLTR",
        "lodl", "perturb", "pg", "dad",
    ]

    # Data
    x_train, Y_train, x_val, Y_val = gen_data()

    # True cost
    xs = np.linspace(0.0, 2.0, 400)
    c_true_xs = true_cost(xs)

    # Layout
    ncols = 4
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    for ax, method in zip(axes, methods):
        entry = best.get(method, {}).get("pg_misspec")
        if entry is None:
            ax.set_visible(False)
            continue
        run_dir = entry["run_dir"]
        slope_int = load_winner_predictor(run_dir)
        if slope_int is None:
            ax.set_title(f"{method} (no ckpt)")
            continue
        slope, intercept = slope_int

        # Scatter
        ax.scatter(x_train, Y_train, s=4, alpha=0.35, color="#888", label="train Y")
        ax.scatter(x_val, Y_val, s=4, alpha=0.35, color="#1f77b4", label="val Y")

        # True cost
        ax.plot(xs, c_true_xs, color="black", lw=1.6, label="true c(x)")
        # Decision boundary (true): vertical at x=0.5
        ax.axvline(0.5, color="black", ls=":", lw=0.8, alpha=0.5)
        # zero line
        ax.axhline(0.0, color="grey", lw=0.5, alpha=0.6)

        # Learned pred
        yhat_xs = intercept + slope * xs
        ax.plot(xs, yhat_xs, color="crimson", lw=1.8,
                label=f"pred (slope={slope:.2f})")

        # Learned threshold
        thresh_str = ""
        if abs(slope) > 1e-9:
            x_thresh = -intercept / slope
            if 0.0 <= x_thresh <= 2.0:
                ax.axvline(x_thresh, color="crimson", ls="--", lw=1.0, alpha=0.7)
                thresh_str = f", x*={x_thresh:.2f}"
            else:
                thresh_str = f", x*={x_thresh:.2f} (oob)"
        else:
            thresh_str = ", slope=0"

        test_regret = entry.get("test", float("nan"))
        ax.set_title(
            f"{method}  test_reg={test_regret:.3f}{thresh_str}",
            fontsize=9
        )
        ax.set_xlim(0, 2)
        ax.set_ylim(-3.0, 3.5)

    # Hide any extra axes
    for ax in axes[len(methods):]:
        ax.set_visible(False)

    fig.suptitle("pg_misspec — Phase 2 winner predictors (yhat = intercept + slope·x)",
                 fontsize=12)
    fig.supxlabel("x")
    fig.supylabel("cost")

    # One legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.96])
    out_png = OUT_DIR / "fig_pg_misspec_predictors.png"
    out_pdf = OUT_DIR / "fig_pg_misspec_predictors.pdf"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
