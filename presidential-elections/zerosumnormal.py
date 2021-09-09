from typing import List
import arviz as az
import numpy as np
import pymc3 as pm
from scipy import stats


if pm.math.erf.__module__.split(".")[0] == "theano":
    import theano
    from theano import tensor as tt
else:
    import aesara as theano
    from aesara import tensor as tt


def extend_axis(array, axis):
    n = array.shape[axis] + 1
    sum_vals = array.sum(axis, keepdims=True)
    norm = sum_vals / (np.sqrt(n) + n)
    fill_val = norm - sum_vals / np.sqrt(n)
    
    out = tt.concatenate([array, fill_val], axis=axis)
    return out - norm


def extend_axis_rev(array, axis):
    if axis < 0:
        axis = axis % array.ndim
    assert axis >= 0 and axis < array.ndim

    n = array.shape[axis]
    last = tt.take(array, [-1], axis=axis)
    
    sum_vals = -last * np.sqrt(n)
    norm = sum_vals / (np.sqrt(n) + n)
    slice_before = (slice(None, None),) * axis
    return array[slice_before + (slice(None, -1),)] + norm


def extend_axis_val(array, axis):
    n = array.shape[axis] + 1
    sum_vals = array.sum(axis, keepdims=True)
    norm = sum_vals / (np.sqrt(n) + n)
    fill_val = norm - sum_vals / np.sqrt(n)
    
    out = np.concatenate([array, fill_val], axis=axis)
    return out - norm


def extend_axis_rev_val(array, axis):
    n = array.shape[axis]
    last = np.take(array, [-1], axis=axis)
    
    sum_vals = -last * np.sqrt(n)
    norm = sum_vals / (np.sqrt(n) + n)
    slice_before = (slice(None, None),) * len(array.shape[:axis])
    return array[slice_before + (slice(None, -1),)] + norm


class ZeroSumTransform(pm.distributions.transforms.Transform):
    name = "zerosum"
    
    _zerosum_axes: List[int]
    
    def __init__(self, zerosum_axes):
        self._zerosum_axes = zerosum_axes
    
    def forward(self, x):
        for axis in self._zerosum_axes:
            x = extend_axis_rev(x, axis=axis)
        return x
    
    def forward_val(self, x, point=None):
        for axis in self._zerosum_axes:
            x = extend_axis_rev_val(x, axis=axis)
        return x
    
    def backward(self, z):
        z = tt.as_tensor_variable(z)
        for axis in self._zerosum_axes:
            z = extend_axis(z, axis=axis)
        return z
    
    def jacobian_det(self, x):
        return tt.constant(0.)
    
    
class ZeroSumNormal(pm.Continuous):
    def __init__(self, sigma=1, *, zerosum_dims=None, zerosum_axes=None, **kwargs):
        shape = kwargs.get("shape", ())
        dims = kwargs.get("dims", None)
        
        if isinstance(shape, int):
            shape = (shape,)
        
        if isinstance(dims, str):
            dims = (dims,)

        self.mu = self.median = self.mode = tt.zeros(shape)
        self.sigma = tt.as_tensor_variable(sigma)
        
        if zerosum_dims is None and zerosum_axes is None:
            if shape:
                zerosum_axes = (-1,)
            else:
                zerosum_axes = ()
        
        if isinstance(zerosum_axes, int):
            zerosum_axes = (zerosum_axes,)
        
        if isinstance(zerosum_dims, str):
            zerosum_dims = (zerosum_dims,)
        
        if zerosum_axes is not None and zerosum_dims is not None:
            raise ValueError("Only one of zerosum_axes and zerosum_dims can be specified.")
        
        if zerosum_dims is not None:
            if dims is None:
                raise ValueError("zerosum_dims can only be used with the dims kwargs.")
            zerosum_axes = []
            for dim in zerosum_dims:
                zerosum_axes.append(dims.index(dim))
        
        self.zerosum_axes = [a if a >= 0 else len(shape) + a for a in zerosum_axes]
        self._degrees_of_freedom = np.prod(
            [s if axis not in zerosum_axes else s - 1 for axis, s in enumerate(shape)]
        )
        self._full_size = np.prod(shape)
        self._rescaling = np.sqrt(self._full_size / self._degrees_of_freedom)
        
        super().__init__(**kwargs, transform=ZeroSumTransform(zerosum_axes))

    def logp(self, x):
        return pm.Normal.dist(sigma=self.sigma / self._rescaling).logp(x)
    
    def _random(self, scale, size):
        samples = stats.norm.rvs(loc=0, scale=scale, size=size)
        for axis in self.zerosum_axes:
            samples -= np.mean(samples, axis=axis, keepdims=True)
        return samples
    
    def random(self, point=None, size=None):
        sigma, scaling = pm.distributions.draw_values(
            [self.sigma, self._rescaling], point=point, size=size
        )
        return pm.distributions.generate_samples(
            self._random, scale=sigma * scaling, dist_shape=self.shape, size=size
        )

    def _distr_parameters_for_repr(self):
        return ["sigma"]

    def logcdf(self, value):
        raise NotImplementedError()
