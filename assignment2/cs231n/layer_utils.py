from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    
    aff_out, aff_cache             = affine_forward(x, w, b)
    batchnorm_out, batchnorm_cache = batchnorm_forward(aff_out, gamma, beta, bn_param)
    relu_out, relu_cache           = relu_forward(batchnorm_out)
    cache                          = (aff_cache, batchnorm_cache, relu_cache)
    return relu_out, cache

def affine_batchnorm_relu_backward(dout, cache):
    
    aff_cache, batchnorm_cache, relu_cache = cache
    
    drelu_input                     = relu_backward(dout, relu_cache)
    dbatchnorm_input, dgamma, dbeta = batchnorm_backward(drelu_input, batchnorm_cache)
    daff_input, dw, db              = affine_backward(dbatchnorm_input, aff_cache)
    
    return daff_input, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache

def conv_batchnorm_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    
    conv_out, conv_cache = conv_forward_fast(x, w, b, conv_param)
    batchnorm_out, batchnorm_cache = spatial_batchnorm_forward(conv_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(batchnorm_out)
    
    cache = (conv_cache, batchnorm_cache, relu_cache)
    
    return relu_out, cache
    
    
def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  #print('the shape of conv_forward_fast_out', a.shape)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache

def conv_relu_conv_relu_pool_forward(x, W1, b1, W2, b2, conv_param, pool_param):
    #print('the shape of x:', x.shape, 'the shape of w1: ', W1.shape, 'w2: ', W2.shape)
    out_first, cache_first = conv_relu_forward(x, W1, b1, conv_param)
    #print('out: ', out_first.shape)
    out_second, cache_second = conv_relu_forward(out_first, W2, b2, conv_param)
    out, pool_cache = max_pool_forward_fast(out_second, pool_param)
    
    cache = (cache_first, cache_second, pool_cache)
    return out, cache

def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_batchnorm_relu_backward(dout, cache):
    
    conv_cache, batchnorm_cache, relu_cache = cache
    
    dout = relu_backward(dout, relu_cache)
    dout_batchnorm, dgamma, dbeta = spatial_batchnorm_backward(dout, batchnorm_cache)
    dx, dw, db = conv_backward_fast(dout_batchnorm, conv_cache)
    
    return dx, dw, db, dgamma, dbeta
    
def conv_relu_conv_relu_pool_backward(dout, cache):
    cache_first, cache_second, cache_pool = cache
    dout_pool = max_pool_backward_fast(dout, cache_pool)
    dout_second, dw2, db2 = conv_relu_backward(dout_pool, cache_second)
    dout, dw1, db1 = conv_relu_backward(dout_second, cache_first)
    
    return dout, dw1, db1, dw2, db2

