"""Эталонные (baseline) распределения и теоретический оптимум по времени отклика.

Используются в экспериментах для сравнения с ЛП-решением:
    * equal_split        — наивный равномерный сплит (x_i = 1/N);
    * round_robin_split  — round-robin (в стационаре эквивалентен равномерному);
    * convex_optimal     — точная минимизация среднего времени отклика (выпуклая
                           задача, решается SciPy); служит нижней границей.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import minimize


def equal_split(mus: List[float], total_lambda: float) -> List[float]:
    """Равномерное распределение потока поровну между всеми серверами."""
    n = len(mus)
    return [total_lambda / n] * n


def round_robin_split(mus: List[float], total_lambda: float) -> List[float]:
    """Round-robin. В стационарном режиме совпадает с равномерным сплитом
    (каждый сервер получает одинаковую долю запросов)."""
    return equal_split(mus, total_lambda)


def convex_optimal(mus: List[float], total_lambda: float, rho_max: float = 0.999) -> List[float]:
    """Минимизация среднего времени отклика W = sum_i lambda_i/(mu_i - lambda_i).

    Это выпуклая (нелинейная) задача. Решается как теоретический ориентир:
    показывает, насколько линейное min-max решение близко к оптимуму по W.
    """
    mus_arr = np.asarray(mus, dtype=float)
    n = len(mus_arr)
    if total_lambda <= 0:
        return [0.0] * n

    def total_wait(lam: np.ndarray) -> float:
        denom = mus_arr - lam
        denom = np.where(denom <= 1e-9, 1e-9, denom)
        return float(np.sum(lam / denom))

    x0 = mus_arr / mus_arr.sum() * total_lambda            # старт: пропорц. mu
    bounds = [(0.0, rho_max * m) for m in mus_arr]
    constraints = {"type": "eq", "fun": lambda lam: float(np.sum(lam) - total_lambda)}

    res = minimize(total_wait, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"ftol": 1e-9, "maxiter": 500})
    lam = np.clip(res.x, 0.0, None)
    # Нормируем к точному суммарному потоку (компенсация численной погрешности).
    if lam.sum() > 0:
        lam = lam * (total_lambda / lam.sum())
    return lam.tolist()
