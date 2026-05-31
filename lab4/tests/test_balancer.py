"""Unit-тесты ЛП-ядра (min-max загрузки) и эталонных распределений."""
import pytest

from core.balancer import solve_min_max
from core.baselines import convex_optimal, equal_split


MUS = [50.0, 30.0, 20.0, 80.0]   # Σμ = 180
TOTAL = 120.0


class TestMinMaxLP:
    def setup_method(self):
        self.alloc = solve_min_max(MUS, TOTAL, rho_max=0.95)

    def test_feasible(self):
        assert self.alloc.feasible is True
        assert self.alloc.status == "Optimal"

    def test_conservation(self):
        # Сумма распределённых потоков равна суммарному входу.
        assert sum(self.alloc.lambdas) == pytest.approx(TOTAL, abs=1e-4)

    def test_fractions_sum_to_one(self):
        assert sum(self.alloc.fractions) == pytest.approx(1.0, abs=1e-6)

    def test_equalizes_utilization(self):
        # При неактивных индивидуальных ограничениях оптимум выравнивает
        # загрузку всех серверов: lambda_i / mu_i = Λ / Σμ.
        expected_rho = TOTAL / sum(MUS)          # 120 / 180 = 0.6667
        rhos = [lam / mu for lam, mu in zip(self.alloc.lambdas, MUS)]
        for r in rhos:
            assert r == pytest.approx(expected_rho, abs=1e-3)
        assert self.alloc.max_rho == pytest.approx(expected_rho, abs=1e-3)

    def test_proportional_to_capacity(self):
        # Следствие: поток пропорционален мощности сервера.
        scale = TOTAL / sum(MUS)
        for lam, mu in zip(self.alloc.lambdas, MUS):
            assert lam == pytest.approx(mu * scale, abs=1e-2)

    def test_min_max_not_worse_than_equal_split(self):
        # max-загрузка ЛП не больше, чем у равномерного сплита.
        eq = equal_split(MUS, TOTAL)
        eq_max_rho = max(lam / mu for lam, mu in zip(eq, MUS))
        assert self.alloc.max_rho <= eq_max_rho + 1e-9


class TestLPInfeasibility:
    def test_overload_is_infeasible(self):
        # Λ выше rho_max * Σμ = 0.95 * 180 = 171 → задача неразрешима.
        alloc = solve_min_max(MUS, total_lambda=200.0, rho_max=0.95)
        assert alloc.feasible is False

    def test_zero_load(self):
        alloc = solve_min_max(MUS, total_lambda=0.0, rho_max=0.95)
        assert alloc.feasible is True
        assert sum(alloc.lambdas) == pytest.approx(0.0)


class TestLPValidation:
    @pytest.mark.parametrize("bad_rho", [0.0, -0.1, 1.5])
    def test_bad_rho_max_raises(self, bad_rho):
        with pytest.raises(ValueError):
            solve_min_max(MUS, TOTAL, rho_max=bad_rho)

    def test_empty_mus_raises(self):
        with pytest.raises(ValueError):
            solve_min_max([], TOTAL)

    def test_nonpositive_mu_raises(self):
        with pytest.raises(ValueError):
            solve_min_max([10.0, 0.0], TOTAL)


class TestConvexOptimal:
    def test_convex_conserves_flow(self):
        lam = convex_optimal(MUS, TOTAL, rho_max=0.95)
        assert sum(lam) == pytest.approx(TOTAL, abs=1e-3)
        assert all(x >= -1e-9 for x in lam)

    def test_convex_not_worse_than_lp_on_response_time(self):
        # Выпуклый оптимум по W не должен уступать ЛП по среднему отклику.
        from core.evaluate import evaluate_allocation

        lp = solve_min_max(MUS, TOTAL, rho_max=0.95)
        cv = convex_optimal(MUS, TOTAL, rho_max=0.95)
        w_lp = evaluate_allocation([f"s{i}" for i in range(4)], MUS, lp.lambdas).avg_w
        w_cv = evaluate_allocation([f"s{i}" for i in range(4)], MUS, cv).avg_w
        assert w_cv <= w_lp + 1e-6
