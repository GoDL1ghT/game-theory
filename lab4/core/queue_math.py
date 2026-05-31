"""Математическое ядро СМО: характеристики очереди M/M/1 (и общий случай M/M/c).

В ServerBalancer каждый сервер моделируется как независимая очередь M/M/1.
Функция :func:`calculate_mm1_metrics` — основная; :func:`calculate_mm_c_metrics`
оставлена как обобщение (M/M/1 — частный случай при c=1) и переиспользована
из исходного кода проекта-примера.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QueueMetrics:
    """Характеристики стационарного режима одной очереди.

    Поля:
        rho:     коэффициент загрузки (lambda / (c * mu)).
        p0:      вероятность простоя (0 заявок в системе).
        p_wait:  вероятность ожидания в очереди (для M/M/1 равна rho).
        l_q:     средняя длина очереди (число ожидающих заявок).
        w_q:     среднее время ожидания в очереди.
        w:       среднее время отклика (ожидание + обслуживание).
        stable:  выполнено ли условие устойчивости rho < 1.
    """

    rho: float
    p0: float
    p_wait: float
    l_q: float
    w_q: float
    w: float
    stable: bool


def calculate_mm1_metrics(lambda_rate: float, mu_rate: float) -> QueueMetrics:
    """Метрики одноканальной СМО M/M/1.

    Аргументы:
        lambda_rate: интенсивность входящего потока (запросов в секунду).
        mu_rate:     интенсивность обслуживания сервера (запросов в секунду).

    Возвращает :class:`QueueMetrics`. При rho >= 1 система неустойчива и
    время отклика бесконечно (очередь растёт неограниченно).
    """
    if mu_rate <= 0:
        raise ValueError("mu должно быть > 0")
    if lambda_rate < 0:
        raise ValueError("lambda должна быть >= 0")

    rho = lambda_rate / mu_rate
    stable = rho < 1.0

    if not stable:
        return QueueMetrics(
            rho=rho, p0=0.0, p_wait=1.0,
            l_q=float("inf"), w_q=float("inf"), w=float("inf"), stable=False,
        )

    p0 = 1.0 - rho                       # вероятность простоя
    p_wait = rho                         # P(ожидание) для M/M/1
    l_q = rho ** 2 / (1.0 - rho)         # средняя длина очереди (Little)
    # Среднее время отклика (W) и ожидания (Wq); при lambda = 0 отклик = 1/mu.
    w = 1.0 / (mu_rate - lambda_rate) if lambda_rate > 0 else 1.0 / mu_rate
    w_q = rho / (mu_rate - lambda_rate) if lambda_rate > 0 else 0.0

    return QueueMetrics(rho=rho, p0=p0, p_wait=p_wait, l_q=l_q, w_q=w_q, w=w, stable=True)


def calculate_mm_c_metrics(lambda_rate: float, mu_rate: float, c: int) -> QueueMetrics:
    """Метрики многоканальной СМО M/M/c (обобщение, c=1 даёт M/M/1).

    Использует формулу Эрланга C для вероятности ожидания. Оставлено для
    полноты модели и совместимости с исходным ядром проекта-примера.
    """
    if c <= 0 or mu_rate <= 0:
        raise ValueError("c и mu должны быть > 0")
    if lambda_rate < 0:
        raise ValueError("lambda должна быть >= 0")

    if c == 1:
        return calculate_mm1_metrics(lambda_rate, mu_rate)

    rho = lambda_rate / (c * mu_rate)
    stable = rho < 1.0
    if not stable:
        return QueueMetrics(
            rho=rho, p0=0.0, p_wait=1.0,
            l_q=float("inf"), w_q=float("inf"), w=float("inf"), stable=False,
        )

    a = lambda_rate / mu_rate                      # предложенная нагрузка (Эрланги)
    sum_terms = sum(a ** n / math.factorial(n) for n in range(c))
    last_term = a ** c / (math.factorial(c) * (1 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    p_wait = last_term * p0                        # формула Эрланга C
    l_q = (rho * p_wait) / (1 - rho)
    w_q = l_q / lambda_rate if lambda_rate > 0 else 0.0
    w = w_q + 1.0 / mu_rate

    return QueueMetrics(rho=rho, p0=p0, p_wait=p_wait, l_q=l_q, w_q=w_q, w=w, stable=stable)
