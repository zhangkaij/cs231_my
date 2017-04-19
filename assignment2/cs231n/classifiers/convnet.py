import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MultiLayerConvet(object):
    """
    A multi-layer convolutional network with the following architecture:
    
    [conv - relu - conv - relu - pool]*N - [affine - relu]*M - affine - softmax
    
    the network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input channels.
    """
    
    def __init__(self, input_dim = (3, 32, 32), num_filters = 32, filter_size = 5,
                 hidden_dim = 100, num_classes = 10, weight_scale = 1e-3, reg = 0.0,
                 num_conv = 2, num_aff = 2, dtype = np.float64):
        """
        Initialize a new network
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        
        self.params = {}
        self.reg    = reg
        self.dtype  = dtype
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_conv = num_conv
        self.num_aff  = num_aff
        
        #为了简洁，设stride = 1, padding = (filter_size -1) / 2，这样每经过一次卷积层，数据的宽和高都不变
        #卷积层初始化
        (C, H, W) = input_dim
        pool_rate = 1 / 4
        out_dim_conv  = num_filters * H * W
        depth = C
        for i in range(num_conv):
            key_w1 = 'W' + str(2 * i + 1)
            key_b1 = 'b' + str(2 * i + 1)
            key_w2 = 'W' + str(2 * i + 2)
            key_b2 = 'b' + str(2 * i + 2)
            self.params[key_w1] = weight_scale * np.random.randn(num_filters, depth, filter_size, filter_size)
            self.params[key_b1] = np.zeros(num_filters)
            depth = num_filters
            self.params[key_w2] = weight_scale * np.random.randn(num_filters, depth, filter_size, filter_size)
            depth = num_filters
            self.params[key_b2] = np.zeros(num_filters)
            
            out_dim_conv = int(out_dim_conv * pool_rate)
        
        #全连接层初始化
        dim_left = out_dim_conv
        for i in range(num_aff):
            dim_right = hidden_dim
            key_w = 'W' + str(i + num_conv * 2 + 1)
            key_b = 'b' + str(i + num_conv * 2 + 1)
            self.params[key_w] = weight_scale * np.random.randn(dim_left, dim_right)
            self.params[key_b] = np.zeros(dim_right)
            dim_left = dim_right
        
        #最后一层初始化
        key_w = 'W' + str(num_conv * 2 + num_aff + 1)
        key_b = 'b' + str(num_conv * 2 + num_aff + 1)        
        self.params[key_w] = weight_scale * np.random.randn(dim_left, num_classes)
        self.params[key_b] = np.zeros(num_classes)
        
        #设置参数
        for k, v in self.params.items():
            #print('init array: ', v.shape)
            self.params[k] = v.astype(dtype)
            
    def loss(self, X, y = None):
    
        conv_param = {'stride': 1, 'pad': int((self.filter_size - 1) / 2)}
        
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        scores = None
        
        
        input  = X
        result = {}
        out    = None
        cache  = None
        
        #卷积层的前向传播
        for i in range(self.num_conv):
            key_w1 = 'W' + str(2 * i + 1)
            key_b1 = 'b' + str(2 * i + 1)
            key_w2 = 'W' + str(2 * i + 2)
            key_b2 = 'b' + str(2 * i + 2)
            w1 = self.params[key_w1]
            b1 = self.params[key_b1]
            w2 = self.params[key_w2]
            b2 = self.params[key_b2]
            
            out, cache = conv_relu_conv_relu_pool_forward(input, w1, b1, w2, b2, conv_param, pool_param)
            
            result[(key_w1, key_b1)] = (out, cache)
            input = out
   
        #全连接层的前向传播    
        for i in range(self.num_aff):
            key_w = 'W' + str(i + self.num_conv * 2 + 1)
            key_b = 'b' + str(i + self.num_conv * 2 + 1)
            
            w = self.params[key_w]
            b = self.params[key_b]
            out, cache   = affine_relu_forward(input, w, b)
            result[(key_w, key_b)] = (out, cache)
            input = out
            
        indice = self.num_aff + 2 * self.num_conv + 1  
        scores, scores_cache = affine_forward(out, self.params['W' + str(indice)], self.params['b' + str(indice)])
        
        if y is None:
            return scores
        
        loss, grads = 0.0, {}
        
        #计算loss
        num_train  = X.shape[0]
    
        exp_scores = np.exp(scores)
        probs      = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    
        correct_logprobs = -np.log(probs[range(num_train), y])
        data_loss = np.sum(correct_logprobs) / num_train
        reg_loss  = 0.0
       # print('reg = ', self.reg)
        for i in range(self.num_conv + self.num_aff + 1):
            key_w = 'W' + str(i + 1)
            reg_loss += 0.5 * self.reg * np.sum(self.params[key_w] * self.params[key_w])
    
        loss = data_loss + reg_loss    
            
        #开始计算grads    
        dscores = probs
        dscores[range(num_train), y] -= 1
        dscores /= num_train   
        
        dout_next, dw_last, db_last = affine_backward(dscores, scores_cache)
        key_w = 'W' + str(indice)
        key_b = 'b' + str(indice)
        grads[key_w] = dw_last + self.reg * self.params[key_w]
        grads[key_b] = db_last
        
        for i in range(self.num_aff)[::-1]:
            key_w = 'W' + str(i + 2 * self.num_conv + 1)
            key_b = 'b' + str(i + 2 * self.num_conv + 1)
            
            _, cache = result[(key_w, key_b)]
            dout_cur, dw, db = affine_relu_backward(dout_next, cache)
            dw += self.reg * self.params[key_w]
            
            grads[key_w] = dw
            grads[key_b] = db
            
            dout_next = dout_cur
            
           
        for i in range(self.num_conv)[::-1]:
            key_w2 = 'W' + str(2 * i + 2)
            key_b2 = 'b' + str(2 * i + 2)
            key_w1 = 'W' + str(2 * i + 1)
            key_b1 = 'b' + str(2 * i + 1)
            
            _, cache = result[(key_w1, key_b1)]
            dout_cur, dw1, db1, dw2, db2 = conv_relu_conv_relu_pool_backward(dout_next, cache)
            dw += self.reg * self.params[key_w]
            
            grads[key_w1] = dw1
            grads[key_b1] = db1
            grads[key_w2] = dw2
            grads[key_b2] = db2
            
            dout_next = dout_cur
            
          
        return loss, grads
          
       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
         
        