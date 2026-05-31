"""Оценка распределения трафика через метрики СМО.

Связывает ЛП-решение (потоки lambda_i) с математическим ядром M/M/1 и считает
агрегированные показатели качества: среднее по системе время отклика, среднюю
вероятность ожидания и устойчивость пула.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.queue_math import calculate_mm1_metrics, QueueMetrics


@dataclass(frozen=True)
class SystemReport:
    """Сводка качества обслуживания при заданном распределении."""

    names: List[str]
    lambdas: List[float]
    per_server: List[QueueMetrics]
    total_lambda: float
    avg_w: float           # среднее время отклика по системе (с)
    avg_p_wait: float      # средняя вероятность ожидания (взвеш. по потоку)
    max_rho: float
    all_stable: bool


def evaluate_allocation(
    names: List[str], mus: List[float], lambdas: List[float]
) -> SystemReport:
    """Считает метрики для распределения потоков по серверам.

    Среднее время отклика взвешивается долей трафика сервера:
        W = sum_i (lambda_i / Lambda) * W_i.
    Если хотя бы один сервер неустойчив (rho_i >= 1), W = inf.
    """
    per_server = [calculate_mm1_metrics(lam, mu) for lam, mu in zip(lambdas, mus)]
    total_lambda = sum(lambdas)
    all_stable = all(m.stable for m in per_server)

    if total_lambda <= 0:
        avg_w = avg_p_wait = 0.0
    elif not all_stable:
        avg_w = float("inf")
        avg_p_wait = sum((lam / total_lambda) * m.p_wait
                         for lam, m in zip(lambdas, per_server))
    else:
        avg_w = sum((lam / total_lambda) * m.w
                    for lam, m in zip(lambdas, per_server))
        avg_p_wait = sum((lam / total_lambda) * m.p_wait
                         for lam, m in zip(lambdas, per_server))

    rhos = [lam / mu for lam, mu in zip(lambdas, mus)]
    max_rho = max(rhos) if rhos else 0.0

    return SystemReport(
        names=list(names), lambdas=list(lambdas), per_server=per_server,
        total_lambda=total_lambda, avg_w=avg_w, avg_p_wait=avg_p_wait,
        max_rho=max_rho, all_stable=all_stable,
    )
