# NOTE: To run these tests remove JIT compilation from powerlaw_psd_gaussian_acc function

import unittest
import numpy as np
import jax
from jax import random

import colorednoise_acc as cn


class TestPowerlawPsdGaussianAcc(unittest.TestCase):

    def test_powerlaw_psd_gaussian_scalar_output_shape(self):
        n = cn.powerlaw_psd_gaussian_acc(1, 16, random_state=random.PRNGKey(1))
        self.assertEqual(n.shape, (16,))

    def test_powerlaw_psd_gaussian_vector_output_shape(self):
        n = cn.powerlaw_psd_gaussian_acc(1, (100, 5), random_state=random.PRNGKey(1))
        self.assertEqual(n.shape, (100, 5))

    def test_powerlaw_psd_gaussian_output_finite(self):
        n = cn.powerlaw_psd_gaussian_acc(1, 16, random_state=random.PRNGKey(1))
        self.assertTrue(np.isfinite(np.asarray(n)).all())

    def test_var_distribution(self):
        # Test deviation from unit variance (with some margin for random errors)
        size = (100, 2**16)
        fmin = 0
        key = random.PRNGKey(1)

        for exponent in [.5, 1, 2]:
            key, subkey = random.split(key)

            y = np.asarray(
                cn.powerlaw_psd_gaussian_acc(
                    exponent, size, random_state=subkey, fmin=fmin
                )
            )

            ystd = y.std(axis=-1)
            var_in = (abs(1 - ystd) < 3 * ystd.std()).mean()

            self.assertTrue(var_in > 0.95)

    def test_small_sample_var(self):
        # Test deviation from unit variance for short time series
        # with odd and even length
        key = random.PRNGKey(1)

        for nsamples in [10, 11]:
            key, subkey = random.split(key)

            ystd = np.asarray(
                cn.powerlaw_psd_gaussian_acc(
                    0, (500, 500, nsamples), random_state=subkey
                )
            ).std(axis=-1)

            self.assertTrue(
                (abs(1 - ystd) < 3 * ystd.std()).mean() > 0.95
            )

    def test_slope_distribution(self):
        # Test deviation from scaling by fitting log-binned spectra
        size = (100, 2**16)
        fmin = 0
        key = random.PRNGKey(1)

        for exponent in [.5, 1, 2]:
            key, subkey = random.split(key)

            y = np.asarray(
                cn.powerlaw_psd_gaussian_acc(
                    exponent, size, random_state=subkey, fmin=fmin
                )
            )

            yfft = np.fft.fft(y)
            f = np.fft.fftfreq(y.shape[-1])
            m = f > 0

            fit, fcov = np.polyfit(
                np.log10(f[m]),
                np.log10(abs(yfft[..., m].T**2)),
                1,
                cov=True
            )

            slope_in = (
                exponent + fit[0] < 3 * np.sqrt(fcov[0, 0])
            ).mean()

            self.assertTrue(slope_in > 0.95)

    def test_cumulative_scaling(self):
        # Check that mean squared displacement for cumulated white noise
        # scales like number of steps as expected for diffusion in 1 dimension

        n_repeats = 1000
        n_steps = 100
        key = random.PRNGKey(1)

        y = np.asarray(
            cn.powerlaw_psd_gaussian_acc(
                0, (n_repeats, n_steps), random_state=key
            )
        )

        mean_squared_displacement = (y.sum(axis=-1)**2).mean(axis=0)
        standard_error = (
            (y.sum(axis=-1)**2).std(axis=0) / np.sqrt(n_repeats)
        )

        self.assertTrue(
            np.abs(n_steps - mean_squared_displacement) < 3 * standard_error
        )

    def test_random_state_type(self):
        exp = 1
        n = 5
        seed = 1

        good_random_states = [
            random.PRNGKey(seed),
        ]

        for random_state in good_random_states:
            cn.powerlaw_psd_gaussian_acc(
                exp, n, random_state=random_state
            )

        bad_random_states = [
            "1",
            0.15,
            [1],
            int(seed),
            None,
            np.random.default_rng(seed),
            np.random.RandomState(seed),
            jax.numpy.array([1, 2, 3]),
        ]

        for random_state in bad_random_states:
            self.assertRaises(
                ValueError,
                cn.powerlaw_psd_gaussian_acc,
                exp,
                n,
                random_state=random_state
            )

    def test_random_state_reproducibility(self):
        exp = 1
        n = 5
        seed = 1

        rs1 = random.PRNGKey(seed)
        rs2 = random.PRNGKey(seed)

        y1 = np.asarray(
            cn.powerlaw_psd_gaussian_acc(
                exp, n, random_state=rs1
            )
        )

        np.random.seed(123)

        y2 = np.asarray(
            cn.powerlaw_psd_gaussian_acc(
                exp, n, random_state=rs2
            )
        )

        np.testing.assert_array_equal(y1, y2)