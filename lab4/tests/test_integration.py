"""Integration-тесты: полный сценарий, REST API (FastAPI) и валидация SimPy."""
import pytest
from fastapi.testclient import TestClient

from api.app import app
from core.balancer import solve_min_max
from core.evaluate import evaluate_allocation
from core.models import Scenario
from core.simulation import simulate_mm1

client = TestClient(app)


VALID_BODY = {
    "servers": [
        {"name": "web-1", "mu": 50.0},
        {"name": "web-2", "mu": 30.0},
        {"name": "web-3", "mu": 20.0},
        {"name": "web-4", "mu": 80.0},
    ],
    "total_lambda": 120.0,
    "rho_max": 0.95,
}


class TestFullScenario:
    def test_scenario_pipeline_stable(self):
        scenario = Scenario(**VALID_BODY)
        alloc = solve_min_max(scenario.mus, scenario.total_lambda, scenario.rho_max)
        report = evaluate_allocation(scenario.names, scenario.mus, alloc.lambdas)
        assert alloc.feasible
        assert report.all_stable
        assert report.avg_w > 0
        # Среднее время отклика должно укладываться в SLA сценария.
        assert report.avg_w * 1000 <= scenario.sla_w_ms

    def test_overloaded_scenario_rejected_by_validation(self):
        body = dict(VALID_BODY, total_lambda=500.0)
        with pytest.raises(ValueError):
            Scenario(**body)


class TestHealthEndpoint:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestBalanceEndpoint:
    def test_balance_success(self):
        resp = client.post("/balance", json=VALID_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["feasible"] is True
        assert data["all_stable"] is True
        assert len(data["servers"]) == 4
        # Доли трафика суммируются в единицу.
        assert sum(s["fraction"] for s in data["servers"]) == pytest.approx(1.0, abs=1e-6)
        # Загрузки выровнены (min-max).
        rhos = [s["rho"] for s in data["servers"]]
        assert max(rhos) - min(rhos) < 1e-2

    def test_balance_overload_returns_422(self):
        body = dict(VALID_BODY, total_lambda=500.0)
        resp = client.post("/balance", json=body)
        assert resp.status_code == 422

    def test_balance_rejects_negative_mu(self):
        body = {
            "servers": [{"name": "a", "mu": -5.0}],
            "total_lambda": 1.0,
            "rho_max": 0.95,
        }
        resp = client.post("/balance", json=body)
        assert resp.status_code == 422


class TestSimulationValidation:
    def test_simulated_matches_analytic_mm1(self):
        # Имитация M/M/1 должна совпасть с аналитикой W = 1/(mu-lambda) = 0.25.
        sim_w = simulate_mm1(lambda_rate=6.0, mu_rate=10.0, n_requests=30000, seed=7)
        analytic_w = 1.0 / (10.0 - 6.0)
        assert sim_w == pytest.approx(analytic_w, rel=0.2)
