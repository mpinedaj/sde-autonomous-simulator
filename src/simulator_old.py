import numpy as np
# ────────────────────────────────────────────────────────────────
#  Fórmulas analíticas exactas de los momentos
# ────────────────────────────────────────────────────────────────
_EPS = 1e-12  # Umbral para considerar un valor ≈ 0
def exact_mean(a, b, X0, t):
    """
    Media exacta E[X_t].
    Se obtiene resolviendo la ODE determinista:
        dm/dt = a·m + b,   m(0) = X₀
    Solución:
        a ≠ 0:  m(t) = (X₀ + b/a)·exp(a·t) − b/a
        a = 0:  m(t) = X₀ + b·t
    """
    t = np.asarray(t, dtype=float)
    if abs(a) < _EPS:
        return X0 + b * t
    eat = np.exp(a * t)
    return (X0 + b / a) * eat - b / a
def _exact_second_moment(a, b, c, d, X0, t):
    """
    Segundo momento exacto E[X_t²].
    Aplicando Itô a X² se obtiene la ODE:
        dM₂/dt = α·M₂ + β·m(t) + d²,   M₂(0) = X₀²
    donde  α = 2a + c²,  β = 2b + 2cd.
    Se resuelve analíticamente con factor integrante, distinguiendo
    los casos según los valores de a y α.
    """
    t = np.asarray(t, dtype=float)
    alpha = 2.0 * a + c ** 2
    beta = 2.0 * b + 2.0 * c * d
    if abs(a) > _EPS:
        # ── Caso a ≠ 0 ──
        P = X0 + b / a               # coeficiente exponencial en m(t)
        A = beta * P                  # coef. de exp(a·t) en g(t)
        C = d ** 2 - beta * b / a     # término constante en g(t)
        diff = a - alpha              # = -(a + c²)
        # I₁ = ∫₀ᵗ exp((a−α)s) ds
        if abs(diff) > _EPS:
            I1 = (np.exp(diff * t) - 1.0) / diff
        else:
            I1 = t.copy() if isinstance(t, np.ndarray) else float(t)
        # I₂ = ∫₀ᵗ exp(−α·s) ds
        if abs(alpha) > _EPS:
            I2 = (1.0 - np.exp(-alpha * t)) / alpha
        else:
            I2 = t.copy() if isinstance(t, np.ndarray) else float(t)
        M2 = np.exp(alpha * t) * (X0 ** 2 + A * I1 + C * I2)
    else:
        # ── Caso a = 0 ──
        # m(t) = X₀ + b·t
        R = beta * X0 + d ** 2        # constante en g(t)
        S = beta * b                   # coef. de t en g(t)
        if abs(alpha) > _EPS:
            # α = c² ≠ 0
            e_neg = np.exp(-alpha * t)
            J1 = R * (1.0 - e_neg) / alpha
            J2 = S * (-t / alpha * e_neg
                      + (1.0 - e_neg) / alpha ** 2)
            M2 = np.exp(alpha * t) * (X0 ** 2 + J1 + J2)
        else:
            # α = 0  ⟹  c = 0 (ruido puramente aditivo)
            M2 = X0 ** 2 + R * t + S * t ** 2 / 2.0
    return M2
def exact_variance(a, b, c, d, X0, t):
    """
    Varianza exacta Var[X_t] = E[X_t²] − (E[X_t])².
    Se calcula a partir de las fórmulas analíticas de la media
    y el segundo momento.
    """
    m = exact_mean(a, b, X0, t)
    M2 = _exact_second_moment(a, b, c, d, X0, t)
    var = M2 - m ** 2
    # Protección numérica: la varianza nunca es negativa
    return np.maximum(var, 0.0)