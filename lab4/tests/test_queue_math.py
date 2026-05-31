"""Unit-тесты математического ядра СМО (M/M/1) против аналитических значений."""
import math

import pytest

from core.queue_math import calculate_mm1_metrics, calculate_mm_c_metrics


class TestMM1Analytic:
    """Проверка формул M/M/1 на эталонном примере: mu=10, lambda=6."""

    def setup_method(self):
        self.m = calculate_mm1_metrics(lambda_rate=6.0, mu_rate=10.0)

    def test_rho(self):
        assert self.m.rho == pytest.approx(0.6)

    def test_p0(self):
        assert self.m.p0 == pytest.approx(0.4)

    def test_p_wait_equals_rho(self):
        # Для M/M/1 вероятность ожидания равна загрузке.
        assert self.m.p_wait == pytest.approx(0.6)

    def test_response_time(self):
        # W = 1 / (mu - lambda) = 1 / 4 = 0.25 c.
        assert self.m.w == pytest.approx(0.25)

    def test_wait_time(self):
        # Wq = rho / (mu - lambda) = 0.6 / 4 = 0.15 c.
        assert self.m.w_q == pytest.approx(0.15)

    def test_queue_length(self):
        # Lq = rho^2 / (1 - rho) = 0.36 / 0.4 = 0.9.
        assert self.m.l_q == pytest.approx(0.9)

    def test_stable(self):
        assert self.m.stable is True


class TestMM1EdgeCases:
    def test_unstable_when_lambda_ge_mu(self):
        m = calculate_mm1_metrics(lambda_rate=10.0, mu_rate=10.0)
        assert m.stable is False
        assert math.isinf(m.w)
        assert math.isinf(m.l_q)

    def test_zero_lambda_gives_service_time(self):
        m = calculate_mm1_metrics(lambda_rate=0.0, mu_rate=10.0)
        assert m.rho == 0.0
        assert m.w == pytest.approx(0.1)   # только обслуживание 1/mu
        assert m.l_q == pytest.approx(0.0)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            calculate_mm1_metrics(lambda_rate=-1.0, mu_rate=10.0)

    def test_nonpositive_mu_raises(self):
        with pytest.raises(ValueError):
            calculate_mm1_metrics(lambda_rate=5.0, mu_rate=0.0)


class TestMMCGeneralization:
    def test_mmc_with_c1_equals_mm1(self):
        # M/M/c при c=1 должна совпадать с M/M/1.
        mm1 = calculate_mm1_metrics(7.0, 10.0)
        mmc = calculate_mm_c_metrics(7.0, 10.0, c=1)
        assert mmc.w == pytest.approx(mm1.w)
        assert mmc.p_wait == pytest.approx(mm1.p_wait)

    def test_mmc_two_channels_stable(self):
        # lambda=15, mu=10, c=2: rho=0.75 < 1, система устойчива.
        m = calculate_mm_c_metrics(15.0, 10.0, c=2)
        assert m.stable is True
        assert 0.0 < m.p_wait < 1.0
        assert m.w > 0
