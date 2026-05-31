"""Имитационная модель M/M/1 на SimPy для валидации аналитических формул.

Используется в тестах и экспериментах: средний по симуляции отклик должен
совпадать с аналитическим W = 1/(mu - lambda) в пределах статистической
погрешности. Это независимая проверка корректности математического ядра.
"""
from __future__ import annotations

import random

import simpy


def simulate_mm1(
    lambda_rate: float,
    mu_rate: float,
    n_requests: int = 20000,
    seed: int = 42,
) -> float:
    """Возвращает среднее время отклика (с) по имитации очереди M/M/1.

    Запросы поступают пуассоновским потоком (экспоненциальные интервалы),
    время обслуживания экспоненциально. Один сервер (один канал).
    """
    if mu_rate <= 0:
        raise ValueError("mu должно быть > 0")
    if lambda_rate <= 0:
        return 1.0 / mu_rate

    rng = random.Random(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    response_times: list[float] = []

    def request(env: simpy.Environment, arrive: float) -> None:
        with server.request() as req:
            yield req
            service = rng.expovariate(mu_rate)
            yield env.timeout(service)
            response_times.append(env.now - arrive)

    def source(env: simpy.Environment) -> None:
        for _ in range(n_requests):
            yield env.timeout(rng.expovariate(lambda_rate))
            env.process(request(env, env.now))

    env.process(source(env))
    env.run()

    # Отбрасываем «прогрев» (первые 10%) для оценки стационарного режима.
    warmup = len(response_times) // 10
    steady = response_times[warmup:]
    return sum(steady) / len(steady) if steady else float("inf")
