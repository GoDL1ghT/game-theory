"""Линейная оптимизация распределения трафика (ЛП-ядро ServerBalancer).

Постановка задачи (min-max загрузки): распределить суммарный поток Λ между
серверами так, чтобы минимизировать максимальную загрузку (узкое горлышко).

    min  z
    s.t. lambda_i <= z * mu_i          для всех i
         sum_i lambda_i = Lambda
         0 <= lambda_i,  0 <= z <= rho_max

Это строго линейная задача (PuLP/CBC решает её точно). Минимизация максимальной
загрузки выравнивает утилизацию серверов и снижает время отклика бутылочного
горлышка; при отсутствии активных индивидуальных ограничений оптимум даёт
классический результат lambda_i ∝ mu_i (все серверы загружены одинаково).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pulp


@dataclass(frozen=True)
class Allocation:
    """Результат распределения трафика."""

    names: List[str]
    mus: List[float]
    lambdas: List[float]      # поток на каждый сервер, запр./с
    fractions: List[float]    # доли трафика x_i (sum = 1)
    max_rho: float            # достигнутая максимальная загрузка z*
    status: str               # статус решателя ("Optimal", "Infeasible", ...)
    feasible: bool


def solve_min_max(
    mus: List[float],
    total_lambda: float,
    rho_max: float = 0.95,
    solver: "pulp.LpSolver | None" = None,
) -> Allocation:
    """Решает ЛП min-max загрузки и возвращает :class:`Allocation`.

    Аргументы:
        mus:          интенсивности обслуживания серверов (запр./с), длины N.
        total_lambda: суммарный входящий поток Λ (запр./с).
        rho_max:      верхняя граница загрузки каждого сервера (0 < rho_max <= 1).
        solver:       решатель PuLP (по умолчанию CBC без логов).

    Исключения:
        ValueError — некорректные входные данные.
    """
    if not mus:
        raise ValueError("Список mu пуст")
    if any(m <= 0 for m in mus):
        raise ValueError("Все mu должны быть > 0")
    if total_lambda < 0:
        raise ValueError("total_lambda должно быть >= 0")
    if not (0 < rho_max <= 1.0):
        raise ValueError("rho_max должно лежать в (0, 1]")

    n = len(mus)
    names = [f"s{i}" for i in range(n)]

    prob = pulp.LpProblem("min_max_load", pulp.LpMinimize)

    # Переменные: поток на сервер lambda_i >= 0 и общий потолок загрузки z.
    lam = [pulp.LpVariable(f"lambda_{i}", lowBound=0) for i in range(n)]
    z = pulp.LpVariable("z", lowBound=0, upBound=rho_max)

    prob += z, "max_utilization"                       # целевая функция

    for i in range(n):
        prob += lam[i] <= z * mus[i], f"load_cap_{i}"  # lambda_i/mu_i <= z
    prob += pulp.lpSum(lam) == total_lambda, "conservation"

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    feasible = status == "Optimal"

    if not feasible:
        return Allocation(
            names=names, mus=list(mus), lambdas=[0.0] * n, fractions=[0.0] * n,
            max_rho=float("nan"), status=status, feasible=False,
        )

    lambdas = [max(0.0, float(v.value() or 0.0)) for v in lam]
    fractions = (
        [li / total_lambda for li in lambdas] if total_lambda > 0 else [1.0 / n] * n
    )
    max_rho = max(li / mu for li, mu in zip(lambdas, mus)) if total_lambda > 0 else 0.0

    return Allocation(
        names=names, mus=list(mus), lambdas=lambdas, fractions=fractions,
        max_rho=max_rho, status=status, feasible=True,
    )
