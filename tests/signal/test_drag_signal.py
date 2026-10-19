"""Test the functionality of the generation of the DRAG corrected signal."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd, vmap

from paraqeet.signal.envelopes import (
    FlatTopGaussianEnvelope,
    GaussEnvelope,
    ZeroEnvelope,
)
from paraqeet.signal.iq_mixer import IQMixer
from paraqeet.signal.signal import DRAGMixer

LEN_SIG = 1001


@pytest.fixture
def time_samples():
    """Generate time samples from the given signal length.

    The given signal length `LEN_SIG` is a module level global variable.

    """
    return np.linspace(0, 10e-9, LEN_SIG)


@pytest.fixture
def gen():
    """Return DRAG signal generator object."""
    tone = GaussEnvelope()
    drag_tone = DRAGMixer(tone)
    return IQMixer(envelopes=[drag_tone])


@pytest.fixture
def zero():
    """Return a zero signal device."""
    return ZeroEnvelope()


@pytest.fixture
def gauss():
    """Return a gaussian signal device."""
    return GaussEnvelope()


@pytest.fixture
def zero_gen():
    """Return a zero tone DRAG generator object."""
    tone = ZeroEnvelope()
    drag_tone = DRAGMixer(tone)
    return IQMixer(envelopes=[drag_tone])


@pytest.fixture
def flattop():
    """Return a FlatTop signal with DRAG."""
    tone = FlatTopGaussianEnvelope()
    drag_tone = DRAGMixer(tone)
    return IQMixer(envelopes=[drag_tone])


@pytest.fixture
def gen_multiple_tones():
    """Return a multiple tone DRAG generator."""
    tone1 = GaussEnvelope()
    params1 = tone1.get_parameters()
    drag_tone1 = DRAGMixer(tone1)

    tone2 = GaussEnvelope()
    params2 = tone2.get_parameters()
    drag_tone2 = DRAGMixer(tone2)
    return (
        IQMixer(envelopes=[drag_tone1, drag_tone2]),
        np.concatenate((params1, params2)),
    )


def test_constant_env(zero_gen, time_samples):
    """Test the values of a DRAG signal using a constant envelope."""
    assert np.all(zero_gen.get_value(time_samples) == np.zeros_like(time_samples))


def test_gen(gen, time_samples) -> None:
    """Computes a sample signal and checks vectorized generation."""
    sig = gen.get_value(time_samples)
    assert len(sig) == LEN_SIG


def test_get_parameters(gen_multiple_tones):
    """Test if the expected amount of parameters is present.

    If multiple tones define the total envelope, the signal needs to have the
    sum of all parameters of the envelope tones plus 4 parameters.

    """
    gen, all_params = gen_multiple_tones
    params = gen.get_parameters()
    assert len(params) == len(all_params) + 4


def test_gradient_shape(gen, time_samples):
    """Test the length of the signal gradient."""
    gen.set_optimizable_parameters(gen.get_parameters())
    _, grads = gen.get_value_and_gradient(time_samples)
    assert grads.shape[0] == time_samples.shape[0]


def test_gradient_flattop(flattop, time_samples):
    """Test the length of the gradient of flattop signal."""
    flattop.set_optimizable_parameters(flattop.get_parameters())
    _, grads = flattop.get_value_and_gradient(time_samples)
    assert grads.shape[0] == time_samples.shape[0]


@pytest.mark.parametrize("env_cls", [GaussEnvelope, FlatTopGaussianEnvelope])
@pytest.mark.parametrize("n_times", [1, 3, 7])
def test_drag_get_time_and_parameter_gradient_values(env_cls, n_times):
    """Check if `get_time_and_parameter_gradient` returns correct (T,P) values."""
    env = env_cls()
    params = env.get_parameters()
    env.set_optimizable_parameters(params)
    times = np.linspace(0, 10e-9, n_times)

    grads = env.get_time_and_parameter_gradient(times)
    assert grads.shape == (n_times, len(params)), f"Expected shape ({n_times}, {len(params)}), got {grads.shape}"

    param_values = [p.get_value() for p in params]
    t_arr = np.array(times, ndmin=1)

    def compute_gradient_vmap(t):
        f = lambda time: jnp.reshape(jnp.stack(env._partial_grads_function(*param_values, time)), (-1,))
        return jacfwd(f)(t)

    ref = vmap(compute_gradient_vmap)(t_arr)
    np.testing.assert_allclose(grads, ref, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("env_cls", [GaussEnvelope, FlatTopGaussianEnvelope])
@pytest.mark.parametrize("n_times", [1, 100])
def test_drag_mixer_gradient_endtoend(env_cls, n_times):
    """Check if DRAGMixer.get_gradient works at T=1."""
    tone = env_cls()
    tone.set_optimizable_parameters(tone.get_parameters())
    drag_tone = DRAGMixer(tone)
    drag_tone.set_optimizable_parameters(tone.get_parameters())
    times = np.linspace(0, 10e-9, n_times)

    grad = drag_tone.get_gradient(times)
    assert grad.shape[0] == n_times, f"Expected {n_times} rows, got {grad.shape[0]}"
    assert np.all(np.isfinite(grad)), "Gradient contains NaN or Inf"
