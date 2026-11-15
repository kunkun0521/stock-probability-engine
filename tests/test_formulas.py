"""文档 §14 单元测试基准 + §14 性质测试。

运行：py -3 -m unittest discover -s tests -v
全部公式都来自 engine/formulas.py 或 engine/models/single.py 的真实实现，不做替身。
"""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import formulas as F                       # noqa: E402
from engine.models.single import VolState, simulate_fhs, stable_seed  # noqa: E402


class TestChapter14(unittest.TestCase):
    def test_normal_tail_probabilities(self):
        mu, sigma = 0.003, 0.02
        self.assertAlmostEqual(F.normal_prob_up(mu, sigma), 0.5596, places=4)
        self.assertAlmostEqual(F.normal_tail_above(mu, sigma, 0.02), 0.1977, places=4)
        self.assertAlmostEqual(F.normal_tail_below(mu, sigma, -0.02), 0.1251, places=4)
        self.assertAlmostEqual(F.normal_tail_above(mu, sigma, 0.04), 0.0322, places=4)
        self.assertAlmostEqual(F.normal_tail_below(mu, sigma, -0.04), 0.0158, places=4)

    def test_bayes_posterior(self):
        self.assertAlmostEqual(F.bayes_posterior(0.7, 0.3, 0.5), 0.70, places=6)

    def test_beta_posterior_mean(self):
        self.assertAlmostEqual(F.beta_posterior_mean(0.5, 20, 10, 8), 0.60, places=6)

    def test_expectation(self):
        self.assertAlmostEqual(F.conditional_expectation(
            [(0.2, 0.05), (0.4, 0.02), (0.4, -0.03)]), 0.006, places=8)

    def test_probability_vs_magnitude(self):
        self.assertAlmostEqual(F.two_way_expectation(0.8, 0.01, -0.05), -0.002, places=8)
        self.assertAlmostEqual(F.two_way_expectation(0.6, 0.05, -0.02), 0.022, places=8)

    def test_z_score(self):
        self.assertAlmostEqual(F.z_score(0.03, 0.002, 0.015), 1.8667, places=4)

    def test_markov_chain(self):
        P = np.array([[0.65, 0.20, 0.15],
                      [0.35, 0.40, 0.25],
                      [0.25, 0.25, 0.50]])
        pi1 = np.array([0.65, 0.20, 0.15])
        pi2 = F.markov_step(pi1, P)
        np.testing.assert_allclose(pi2, [0.53, 0.2475, 0.2225], atol=1e-8)
        np.testing.assert_allclose(F.markov_horizon(pi1, P, 2), pi2, atol=1e-8)

    def test_binomial(self):
        self.assertAlmostEqual(F.binomial_pmf(5, 3, 0.6), 0.3456, places=6)

    def test_gbm_monte_carlo(self):
        rng = np.random.default_rng(20260101)
        z = rng.standard_normal(100000)
        S = F.gbm_step(np.full(100000, 100.0), 0.10, 0.30, 20 / 244, z)
        self.assertAlmostEqual(float(S.mean()), 100.82, delta=0.3)
        self.assertAlmostEqual(F.gbm_terminal_mean(100, 0.10, 0.30, 20 / 244),
                               float(S.mean()), delta=0.3)

    def test_garch_variance_recursion(self):
        om, al, be, var_next = 0.05, 0.05, 0.90, 2.0
        self.assertAlmostEqual(om / (1 - al - be), 1.0, places=8)
        bar = om / (1 - al - be)
        self.assertAlmostEqual(bar + (al + be) * (var_next - bar), 1.95, places=8)
        self.assertAlmostEqual(F.garch_h_variance(om, al, be, var_next, 2), 3.95, places=8)

    def test_gpd_tail(self):
        var, es = F.gpd_var_es(2.0, 1.0, 0.1, 1000, 100, 0.99)
        self.assertAlmostEqual(var, 4.589, delta=0.01)
        self.assertAlmostEqual(es, 5.988, delta=0.01)

    def test_wilson_and_pvalue_helpers(self):
        lo, hi = F.wilson_interval(8, 10)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        self.assertGreaterEqual(F.kupiec_pvalue(5, 100, 0.95), 0.0)
        self.assertAlmostEqual(F.binomial_winrate_pvalue(60, 100, 0.5),
                               float(stats.binomtest(60, 100, 0.5).pvalue), places=8)


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(
            quantile_taus=[0.05, 0.25, 0.5, 0.75, 0.95],
            consistency_mode="direction", consistency_iters=24,
            drop_threshold_1=0.05, drop_threshold_2=0.10,
            rise_threshold_1=0.05, rise_threshold_2=0.10,
        )

    def _vol(self):
        v = VolState()
        v.model = "EWMA"
        v.sigma_daily = np.full(10, 0.02, dtype=np.float32)
        v.var_by_h = np.array([(0.02 ** 2) * h for h in range(1, 11)])
        v.residual_pool = np.array([0.0], dtype=np.float32)
        return v

    def test_probability_sum_and_quantile_monotonic(self):
        rng = np.random.default_rng(7)
        z = rng.standard_normal((10000, 10))
        for p in (0.35, 0.5, 0.72):
            s = simulate_fhs(self._vol(), 5, p, self.cfg, z, 0.10)
            self.assertAlmostEqual(s.p_up + s.p_down, 1.0, places=8)
            self.assertTrue(np.all(np.diff(s.q) >= 0), "分位数必须单调不减")
            self.assertGreaterEqual(s.var99, s.var95)
            for v in (s.prob_drop_5, s.prob_drop_10, s.prob_rise_5, s.prob_rise_10,
                      s.var95, s.var99, s.cvar95):
                self.assertTrue(math.isfinite(v))

    def test_consistency_direction(self):
        rng = np.random.default_rng(11)
        z = rng.standard_normal((10000, 10))
        for h in (1, 3, 5, 10):
            for p in (0.3, 0.55, 0.8):
                s = simulate_fhs(self._vol(), h, p, self.cfg, z, 0.10)
                self.assertLessEqual(abs(s.sim_p_up - p), 0.002,
                                     f"h={h} p={p} 模拟 P(R>0)={s.sim_p_up}")

    def test_fixed_seed_reproducible(self):
        from datetime import date
        self.assertEqual(stable_seed("600519", date(2026, 11, 13), 20260101),
                         stable_seed("600519", date(2026, 11, 13), 20260101))
        rng1 = np.random.default_rng(stable_seed("600519", date(2026, 11, 13), 20260101))
        rng2 = np.random.default_rng(stable_seed("600519", date(2026, 11, 13), 20260101))
        np.testing.assert_array_equal(rng1.standard_normal(100), rng2.standard_normal(100))

    def test_risk_grade_and_bss(self):
        self.assertEqual(F.risk_level_from_pctile(0.1), 1)
        self.assertEqual(F.risk_level_from_pctile(0.5), 2)
        self.assertEqual(F.risk_level_from_pctile(0.8), 3)
        self.assertEqual(F.risk_level_from_pctile(0.95), 4)
        self.assertEqual(F.grade_from_period(0.05, 0.01), "A")
        self.assertEqual(F.grade_from_period(0.05, -0.01), "B")
        self.assertEqual(F.grade_from_period(-0.02, -0.1), "C")
        self.assertEqual(F.grade_from_period(0.05, 0.01, 5), "D")
        y = np.array([1, 0, 1, 0])
        p = np.array([0.7, 0.3, 0.6, 0.4])
        self.assertGreater(F.brier_skill_score(y, p, 0.5), 0)

    def test_quantile_loss_and_portfolio_metrics(self):
        u = np.array([-1.0, 1.0])
        self.assertAlmostEqual(F.pinball_loss(u, 0.5), 0.5, places=8)
        nav = np.array([1.0, 1.01, 1.005, 1.02])
        m = F.portfolio_metrics(nav)
        self.assertGreaterEqual(m["maxDrawdown"], 0)
        self.assertTrue(math.isfinite(m["sharpe"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
