import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_current_model_milp import LinearModel, replay_request, solve_agent


PARAMS = {
    "battery_cap": 100.0, "raw_ch": 0.98, "raw_dis": 0.98, "dg_max": 225.0,
    "costa": 0.0083, "costb": 5.76, "costc": 64.0,
    "bessa": 0.0152, "bessb": 5.61, "bessc": 24.0,
}


def simulate(net, prices, pds, requested, initial):
    soc, total = initial, 0.0
    actual, states = [], [soc]
    for demand, price, pd, q in zip(net, prices, pds, requested):
        before = soc
        soc -= 0.98 * q / 100
        pb = q
        if soc < 0.2:
            soc, pb = 0.2, (before - 0.2) * 100
        if soc > 0.8:
            soc, pb = 0.8, (before - 0.8) * 100
        x = pb + 300 * (1 - soc)
        cost = (0.0152 * x**2 + 5.61 * x + 24) * (0.3 if pb > 0 else 1)
        cost += 0.5 * (0.0083 * pd**2 + 5.76 * pd + 64)
        cost += max(demand - pd - pb, 0) * price
        total += cost
        actual.append(pb)
        states.append(soc)
    return total, actual, states


class CurrentModelMilpTest(unittest.TestCase):
    def test_all_five_transition_modes_match_reference_projection(self):
        original = LinearModel.solve
        for mode in range(5):
            with self.subTest(mode=mode):
                def force_mode(model, gap, limit):
                    choices = next(
                        list(entries) for entries in model.rows
                        if len(entries) == 5 and all(
                            model.integer[i] and value == 1 for i, value in entries.items()
                        )
                    )
                    model.row({choices[mode]: 1}, 1, 1)
                    return original(model, gap, limit)

                initial = 0.2 if mode == 3 else 0.35
                with patch.object(LinearModel, "solve", force_mode):
                    result = solve_agent([1], [-1], PARAMS, initial, segments=16, time_limit=5)
                s = result["schedule"]
                self.assertEqual(s["mode"], [mode])
                q = replay_request(
                    initial, s["soc"][1], s["requested_pb"][0], mode, 100, 0.98, 0.98,
                )
                _, actual, states = simulate([1], [-1], s["pd"], [q], initial)
                np.testing.assert_allclose(s["pb"], actual, atol=2e-4)
                np.testing.assert_allclose(s["soc"], states, atol=2e-6)

    def test_projection_import_and_cost_certificate(self):
        net, prices = np.array([100, 0, -100, 1]), np.array([-2, 2, -2, 1])
        result = solve_agent(net, prices, PARAMS, 0.35, segments=20, time_limit=8)
        s = result["schedule"]
        before = 0.35
        requested = []
        for t in range(len(net)):
            q = replay_request(
                before, s["soc"][t + 1], s["requested_pb"][t], s["mode"][t],
                100, 0.98, 0.98,
            )
            requested.append(q)
            _, _, next_states = simulate([0], [0], [0], [q], before)
            before = next_states[-1]
        exact, actual, states = simulate(net, prices, s["pd"], requested, 0.35)
        np.testing.assert_allclose(s["soc"], states, atol=2e-6)
        np.testing.assert_allclose(s["pb"], actual, atol=2e-4)
        np.testing.assert_allclose(s["pg"], net - np.array(s["pd"]) - actual, atol=2e-4)
        error = result["pwl_objective"] - exact
        self.assertGreaterEqual(error, -1e-3)
        self.assertLessEqual(error, result["quadratic_interpolation_error_bound"] + 1e-3)
        self.assertLessEqual(result["restricted_domain_exact_cost_lower_bound"], exact + 1e-3)
        self.assertLess(result["max_constraint_violation"], 2e-5)

    def test_negative_prices_cannot_manufacture_unlimited_import(self):
        result = solve_agent([1], [-2], PARAMS, 0.8, segments=20, time_limit=5)
        s = result["schedule"]
        self.assertLessEqual(max(s["pg"]), 101 + 1e-6)
        exact, _, _ = simulate([1], [-2], s["pd"], s["requested_pb"], 0.8)
        self.assertGreater(exact, -202)

    def test_discharge_discount_remains_discontinuous(self):
        result = solve_agent([0], [1], PARAMS, 0.8, segments=20, time_limit=5)
        s = result["schedule"]
        self.assertGreater(s["pb"][0], 0)
        self.assertLess(s["pb"][0], 0.01)
        _, _, states = simulate([0], [1], s["pd"], s["requested_pb"], 0.8)
        self.assertGreater(states[-1], 0.799)

    def test_approximation_validation(self):
        with self.assertRaises(ValueError):
            solve_agent([0], [1], PARAMS, 0.35, segments=1)
        with self.assertRaises(ValueError):
            solve_agent([0], [1], PARAMS, 0.1)


if __name__ == "__main__":
    unittest.main()
