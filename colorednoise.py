"""Generate colored noise, accelerated with JAX."""

from typing import Union, Iterable, Optional
import jax
import jax.numpy as jnp
from jax import random
from jax.numpy.fft import irfft, rfftfreq

import numpy as np

from functools import partial

@partial(jax.jit, static_argnums=(1,))
def powerlaw_psd_gaussian(
    exponent: float, 
    size: Union[int, Iterable[int]], 
    fmin : float = 0,
    random_state: Optional[Union[int, random.PRNGKey]] = None
    ):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
    The power-spectrum of the generated noise is proportional to

    S(f) = (1 / f)**beta
    flicker / pink noise:   exponent beta = 1
    brown noise:            exponent beta = 2

    Furthermore, the autocorrelation decays proportional to lag**-gamma
    with gamma = 1 - beta for 0 < beta < 1.
    There may be finite-size issues for beta close to one.

    shape : int or iterable
    The output has the given shape, and the desired power spectrum in
    the last coordinate. That is, the last dimension is taken as time,
    and all other components are independent.

    fmin : float, optional
    Low-frequency cutoff.
    Default: 0 corresponds to original paper. 
    NOTE : fmin is not used here, default fmin = 0. this is done to allow just in time compilation
    
    The power-spectrum below fmin is flat. fmin is defined relative
    to a unit sampling rate (see numpy's rfftfreq). For convenience,
    the passed value is mapped to max(fmin, 1/samples) internally
    since 1/samples is the lowest possible finite frequency in the
    sample. The largest possible value is fmin = 0.5, the Nyquist
    frequency. The output for this value is white noise.

    random_state :  int, jax.random.PRNGKey, optional
    Optionally sets the state of JAX's underlying random number generator.
    Integer-compatible values or None are passed to jax.random.PRNGKey.
    jax.random.PRNGKey is used directly.
    Default: None.

    Returns
    -------
    out : array
    The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, int):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we assume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)
    
    # NOTE : fmin is not used here, default fmin = 0
    # # Validate / normalise fmin
    # if 0 <= fmin <= 0.5:
    #     fmin = max(fmin, 1./samples) # Low frequency cutoff
    # else:
    #     raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    # Build scaling factors for all frequencies
    s_scale = f    
    s_scale = s_scale.at[0].set(s_scale[1])
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.) # correct f = +-0.5
    sigma = 2 * jnp.sqrt(jnp.sum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(jnp.newaxis,) * dims_to_add + (Ellipsis,)]
    
    # prepare random number generator
    print(type(random_state))
    if random_state is None or isinstance(random_state, int):
        random_state = random.PRNGKey(random_state if random_state is not None else 0)
    elif isinstance(random_state, jax.Array):
        pass
    else :
        raise ValueError(
            "random_state must be one of integer, random.PRNGKey, "
            "or None.")
    
    # Generate scaled random power + phase
    sr = random.normal(random_state, shape=size) * s_scale
    si = random.normal(random_state, shape=size) * s_scale
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * jnp.sqrt(2))    # Fix magnitude
    
    # Regardless of signal length, the DC component must be real
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * jnp.sqrt(2))    # Fix magnitude
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    
    return y
