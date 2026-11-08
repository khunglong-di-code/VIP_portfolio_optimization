"""Kiểm tra hợp đồng FBM: python -m unittest -v test_fbm.

Các bài toán dùng vector float64 shape (2,); solver trả dictionary README.
"""

import unittest

import numpy as np

from algorithms import fbm


class TestFBM(unittest.TestCase):
    def test_known_l1_solution(self):
        # min 0.5*||x-b||^2 + lam*||x||_1 có nghiệm soft-threshold(b).
        b = np.array([3., -0.2])
        lam = 0.5
        x0 = np.array([-4., 5.])
        before = x0.copy()

        def B(x):
            return x - b

        def prox(y, gamma):
            return np.sign(y) * np.maximum(np.abs(y) - gamma * lam, 0.)

        result = fbm(x0, B, prox, gamma=0.5, tol=1e-10)
        np.testing.assert_allclose(result['x'], [2.5, 0.], atol=1e-9)
        np.testing.assert_array_equal(x0, before)
        self.assertTrue(result['converged'])
        self.assertEqual(result['stop_reason'], 'tolerance')
        self.assertLessEqual(result['residual'], 1e-10)

    def test_iteration_limit_and_snapshots(self):
        x0 = np.array([4., -2.])
        result = fbm(x0, lambda x: x, lambda y, gamma: y,
                     gamma=0.5, max_iter=3, tol=1e-12,
                     iter_list=iter([1, 2, 3, 10]))
        self.assertFalse(result['converged'])
        self.assertEqual(result['stop_reason'], 'max_iter')
        self.assertEqual(result['iterations'], 3)
        self.assertEqual(set(result['res']), {0, 1, 2, 3})
        for k, snapshot in result['res'].items():
            np.testing.assert_array_equal(snapshot, x0 * 0.5**k)
        self.assertAlmostEqual(result['residual'], np.linalg.norm(x0) / 8)
        result['x'][0] = 99.
        result['res'][1][0] = 88.
        np.testing.assert_array_equal(result['res'][0], x0)
        np.testing.assert_array_equal(result['res'][3], x0 / 8)

    def test_early_stop_does_not_invent_snapshots(self):
        result = fbm(np.zeros(2), lambda x: x, lambda y, gamma: y,
                     gamma=1., iter_list=(1, 10))
        self.assertEqual(result['iterations'], 1)
        self.assertEqual(set(result['res']), {0, 1})
        self.assertEqual(result['residual'], 0.)

    def test_invalid_operator_outputs(self):
        for bad in (1., np.ones((2, 1)), np.array([np.nan, 0.]),
                    np.array([np.inf, 0.])):
            with self.subTest(output=bad):
                with self.assertRaises(ValueError):
                    fbm(np.ones(2), lambda x: bad, lambda y, gamma: y, 1.)
                with self.assertRaises(ValueError):
                    fbm(np.ones(2), lambda x: x, lambda y, gamma: bad, 1.)

    def test_invalid_parameters(self):
        for kwargs in ({'gamma': 0.}, {'gamma': np.inf}, {'tol': np.nan},
                       {'tol': -1.}, {'max_iter': 0}, {'max_iter': 1.5},
                       {'iter_list': [1.5]}, {'iter_list': [-1]},
                       {'x0': np.ones((2, 1))}, {'x0': []},
                       {'x0': [np.nan, 1.]}):
            arguments = dict(x0=np.ones(2), B=lambda x: x,
                             prox=lambda y, gamma: y, gamma=1.)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fbm(**arguments)


if __name__ == '__main__':
    unittest.main()
