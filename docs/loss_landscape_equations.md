# Surrogate Loss Equations

Let $\hat{c}$ = predicted costs, $c$ = true costs, $z^* = \arg\min_{z \in \mathcal{Z}} c^\top z$, $V(c) = \min_{z \in \mathcal{Z}} c^\top z$.

---

**Decision Regret** (true task loss, piecewise constant, not differentiable):
$$\ell(\hat{c}, c) = c^\top \hat{z}(\hat{c}) - V(c), \quad \hat{z}(\hat{c}) = \arg\min_{z \in \mathcal{Z}} \hat{c}^\top z$$

**SPO+** (convex upper bound, $\geq \ell$):
$$\mathcal{L}_\text{SPO+}(\hat{c}, c) = -V(2\hat{c} - c) + 2\hat{c}^\top z^* - V(c)$$

**PGB** — Perturbation Gradient Backward (upper bound, $\geq \ell$):
$$\mathcal{L}_\text{PGB}(\hat{c}, c) = \frac{V(\hat{c}) - V(\hat{c} - \sigma c)}{\sigma} - V(c)$$

**DBB** — Decision-Focused Black-Box (lower bound, $\leq \ell$):
$$\mathcal{L}_\text{DBB}(\hat{c}, c) = \frac{V(\hat{c} + \sigma c) - V(\hat{c})}{\sigma} - V(c)$$

**Perturb / DPO** (smooth, no structural bound):
$$\mathcal{L}_\text{DPO}(\hat{c}, c) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\!\left[c^\top \hat{z}(\hat{c} + \sigma\varepsilon)\right] - V(c)$$

**LODL** (learned surrogate, no structural guarantee):
$$\mathcal{L}_\text{LODL}(\hat{c}, c) = f_\phi(\hat{c}, c), \quad \phi \text{ trained to mimic regret}$$
