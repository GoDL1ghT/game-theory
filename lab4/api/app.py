"""REST-интерфейс ServerBalancer на FastAPI.

Эндпоинты:
    GET  /health   — health-check для контейнера/оркестратора.
    POST /balance  — принимает пул серверов и поток, возвращает ЛП-распределение
                     и метрики СМО.

Запуск: ``uvicorn api.app:app --host 0.0.0.0 --port 8000``.
"""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.balancer import solve_min_max
from core.evaluate import evaluate_allocation
from core.models import Scenario, ServerSpec

app = FastAPI(
    title="ServerBalancer",
    description="Прототип балансировщика нагрузки (СМО + ЛП)",
    version="1.0.0",
)


class BalanceRequest(BaseModel):
    """Тело запроса на балансировку."""

    servers: List[ServerSpec]
    total_lambda: float = Field(..., ge=0)
    rho_max: float = Field(0.95, gt=0, le=1.0)


class ServerResult(BaseModel):
    name: str
    mu: float
    lambda_assigned: float
    fraction: float
    rho: float
    w_ms: float
    p_wait: float
    stable: bool


class BalanceResponse(BaseModel):
    status: str
    feasible: bool
    max_rho: float
    avg_w_ms: float
    avg_p_wait: float
    all_stable: bool
    servers: List[ServerResult]


@app.get("/health")
def health() -> dict:
    """Проверка живости сервиса."""
    return {"status": "ok", "service": "server-balancer"}


@app.post("/balance", response_model=BalanceResponse)
def balance(req: BalanceRequest) -> BalanceResponse:
    """Считает оптимальное распределение трафика и метрики СМО."""
    # Валидация через Pydantic-сценарий (проверка мощности пула и пр.).
    try:
        Scenario(servers=req.servers, total_lambda=req.total_lambda, rho_max=req.rho_max)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mus = [s.mu for s in req.servers]
    names = [s.name for s in req.servers]

    alloc = solve_min_max(mus, req.total_lambda, req.rho_max)
    if not alloc.feasible:
        raise HTTPException(status_code=409, detail=f"ЛП не решено: {alloc.status}")

    report = evaluate_allocation(names, mus, alloc.lambdas)

    servers = [
        ServerResult(
            name=names[i], mu=mus[i], lambda_assigned=alloc.lambdas[i],
            fraction=alloc.fractions[i], rho=report.per_server[i].rho,
            w_ms=report.per_server[i].w * 1000.0,
            p_wait=report.per_server[i].p_wait, stable=report.per_server[i].stable,
        )
        for i in range(len(names))
    ]

    return BalanceResponse(
        status=alloc.status, feasible=alloc.feasible, max_rho=alloc.max_rho,
        avg_w_ms=report.avg_w * 1000.0, avg_p_wait=report.avg_p_wait,
        all_stable=report.all_stable, servers=servers,
    )
