```rst
colorednoise.py
===============

Generate Gaussian distributed noise with a power law spectrum with arbitrary
exponents.

An exponent of two corresponds to Brownian noise. Smaller exponents
yield long-range correlations, i.e. pink noise for an exponent of 1 (also
called 1/f noise or flicker noise).

Based on the algorithm in:

    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

Further reading:
`Colors of noise on Wikipedia <https://en.wikipedia.org/wiki/Colors_of_noise>`_


JAX-Accelerated Colored Noise Generation
----------------------------------------

This fork extends the original ``colorednoise`` library with a JAX-based
implementation of the Timmer-Koenig algorithm for generating Gaussian
colored noise.

The extension was developed to support efficient colored-noise generation
in probabilistic modeling and simulation applications, particularly for
continuum robot motion-model learning.

The function ``powerlaw_psd_gaussian_acc()`` uses JAX operations and
just-in-time (JIT) compilation to enable accelerated execution on
supported hardware.

The original NumPy implementation, ``powerlaw_psd_gaussian()``, remains
available and unchanged.

**Main modifications:**

- Replaced NumPy array operations with JAX equivalents.
- Introduced JIT compilation using ``jax.jit``.
- Implemented random-number generation using explicit JAX PRNG keys.
- Adapted FFT computations and array updates to JAX.
- Restricted the accelerated implementation to the default frequency
  cutoff (``fmin=0``).

**Example:**

.. code:: python

   import jax
   import colorednoise as cn

   key = jax.random.PRNGKey(42)

   beta = 1.0
   samples = 1000

   key, subkey = jax.random.split(key)
   noise1 = cn.powerlaw_psd_gaussian_acc(
       beta, samples, random_state=subkey
   )

   key, subkey = jax.random.split(key)
   noise2 = cn.powerlaw_psd_gaussian_acc(
       beta, samples, random_state=subkey
   )

The two calls generate different noise realizations while remaining
reproducible when initialized with the same seed.

For multidimensional noise generation, specify the output shape as a tuple:

.. code:: python

   key, subkey = jax.random.split(key)

   noise = cn.powerlaw_psd_gaussian_acc(
       exponent=1.0,
       size=(80, 1000),
       random_state=subkey
   )

This generates 80 colored-noise time series, each containing 1000 samples.

**Note:** The accelerated implementation currently supports only the
default frequency cutoff (``fmin=0``). For nonzero ``fmin`` values,
use the original NumPy implementation.


Installation
------------

Clone this repository and install it locally:

.. code:: bash

   pip install -e .


Dependencies
------------

Original implementation:

- NumPy >= 1.17.0

JAX-accelerated implementation:

- JAX
- JAXlib

The original implementation supports Python >= 3.6.15.
The JAX-accelerated implementation requires a Python version compatible
with the installed JAX release.

For Python 2, please use colorednoise version 1.x.


Examples
--------

Generate a single colored-noise time series:

.. code:: python

   import colorednoise as cn

   beta = 1  # the exponent
   samples = 2**18  # number of samples to generate

   y = cn.powerlaw_psd_gaussian(beta, samples)

   # Optionally plot the Power Spectral Density with Matplotlib
   # from matplotlib import mlab
   # from matplotlib import pylab as plt
   # s, f = mlab.psd(y, NFFT=2**13)
   # plt.loglog(f, s)
   # plt.grid(True)
   # plt.show()


Generate several time series of independent identically distributed
variables, repeating the simulation of each variable multiple times:

.. code:: python

   import colorednoise as cn

   n_repeats = 10    # repeat simulations
   n_variables = 5   # independent variables in each simulation
   timesteps = 1000  # number of timesteps for each variable

   y = cn.powerlaw_psd_gaussian(
       1, (n_repeats, n_variables, timesteps)
   )

   # The expected variance for each variable is 1,
   # but each realization is different.
   print(y.std(axis=-1))


Generate a broken power law spectrum, white below a specified frequency:

.. code:: python

   import colorednoise as cn
   from matplotlib import mlab
   from matplotlib import pylab as plt

   y = cn.powerlaw_psd_gaussian(1, 10**5, fmin=.05)

   s, f = mlab.psd(y, NFFT=2**9)

   # plt.loglog(f, s)
   # plt.grid(True)
   # plt.show()
```