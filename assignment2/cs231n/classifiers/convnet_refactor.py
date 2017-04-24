import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MultiLayerConvet_refactor(object):
    """
    A multi-layer convolutional network with the following architecture:
    
    [conv - relu - conv - relu - pool]*N - [affine - relu]*M - affine - softmax
    
    the network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input channels.
    """
    
    def __init__(self, input_dim = (3, 32, 32), num_filters = 32, filter_size = 5,
                 hidden_dim = 100, num_classes = 10, weight_scale = 1e-3, reg = 0.0,
                 num_conv = 4, num_aff = 1, dtype = np.float64):
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
            indice = str(i + 1)
                       
            self.params['W' + indice] = weight_scale * np.random.randn(num_filters, depth, filter_size, filter_size)
            self.params['b' + indice] = np.zeros(num_filters)
            self.params['gamma' + indice] = np.ones(num_filters)
            self.params['beta' + indice]  = np.zeros(num_filters)            
            
            depth = num_filters
            
            #每隔两层添加一个汇聚层
            if i % 2 == 1:               
                out_dim_conv = int(out_dim_conv * pool_rate)
        
        #全连接层初始化
        dim_left = out_dim_conv
        for i in range(num_aff):
            
            dim_right = hidden_dim
            indice = str(i + num_conv + 1)
            
            self.params['W' + indice] = weight_scale * np.random.randn(dim_left, dim_right)
            self.params['b' + indice] = np.zeros(dim_right)
            self.params['gamma' + indice] = np.ones(dim_right)
            self.params['beta' + indice] = np.zeros(dim_right)
            dim_left = dim_right
        
        #最后一层初始化
        key_w = 'W' + str(num_conv + num_aff + 1)
        key_b = 'b' + str(num_conv + num_aff + 1)        
        self.params[key_w] = weight_scale * np.random.randn(dim_left, num_classes)
        self.params[key_b] = np.zeros(num_classes)
        
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in range(self.num_conv + self.num_aff)]
            
        #设置参数
        for k, v in self.params.items():
            #print('init array: ', v.shape)
            self.params[k] = v.astype(dtype)
    
    def softmaxOut(self, scores, y):
        
         #计算loss
        num_train  = scores.shape[0]
    
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
        
        return loss, dscores
    
    def loss(self, X, y = None):
    
        mode = 'test' if y is None else 'train'
        
        conv_param = {'stride': 1, 'pad': int((self.filter_size - 1) / 2)}        
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        for bn_param in self.bn_params:
            bn_param['mode'] = mode
        
        scores = None
        num_layer = self.num_conv + self.num_aff
        
        input  = X
        result = {}
        out    = None
        cache  = None
        
        #卷积层的前向传播
        for i in range(self.num_conv):
            
            indice = str(i + 1)        
            w = self.params['W' + indice]
            b = self.params['b' + indice]
            gamma = self.params['gamma' + indice]
            beta  = self.params['beta' + indice]
            
            out, cache = conv_batchnorm_relu_forward(input, w, b, gamma, beta, conv_param, self.bn_params[i])
            #out, cache = conv_relu_conv_relu_pool_forward(input, w1, b1, w2, b2, conv_param, pool_param)
            
            result[('W' + indice, 'b' + indice)] = (out, cache)
            if i % 2:
                out, pool_cache = max_pool_forward_fast(out, pool_param)
                result['pool' + str(int(i / 2))] = (out, pool_cache)
                
            input = out
   
        #全连接层的前向传播    
        for i in range(self.num_aff):
            
            indice = str(i + self.num_conv + 1)          
            w = self.params['W' + indice]
            b = self.params['b' + indice]
            gamma = self.params['gamma' + indice]
            beta  = self.params['beta'  + indice]
            
            out, cache   = affine_batchnorm_relu_forward(input, w, b, gamma, beta, self.bn_params[i + self.num_conv])
            result[('W' + indice, 'b' + indice)] = (out, cache)
            input = out
            
        last_indice = str(self.num_aff + self.num_conv + 1)  
        scores, scores_cache = affine_forward(out, self.params['W' + last_indice], self.params['b' + last_indice])
        
        if y is None:
            return scores
        
        loss, grads = 0.0, {}
        
        loss, dscores = self.softmaxOut(scores, y)
                
        dout_next, dw_last, db_last = affine_backward(dscores, scores_cache)
        
        grads['W' + last_indice] = dw_last + self.reg * self.params['W' + last_indice]
        grads['b' + last_indice] = db_last
        
        for i in range(self.num_aff)[::-1]:
            indice = str(i + self.num_conv + 1)
            
            _, cache = result[('W' + indice, 'b' + indice)]
            dout_cur, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout_next, cache)
            dw += self.reg * self.params['W' + indice]
            
            grads['W' + indice] = dw
            grads['b' + indice] = db
            grads['gamma' + indice] = dgamma
            grads['beta' + indice]  = dbeta
            dout_next = dout_cur
            
           
        for i in range(self.num_conv)[::-1]:
            
            indice = str(i + 1)
            
            if i % 2:
                _, cache = result['pool' + str(int(i / 2))]
                dout_next = max_pool_backward_fast(dout_next, cache)
           
            _, cache = result[('W' + indice, 'b' + indice)]
            dout_cur, dw, db, dgamma, dbeta = conv_batchnorm_relu_backward(dout_next, cache)
            dw += self.reg * self.params['W' + indice]
            
            grads['W' + indice] = dw
            grads['b' + indice] = db
            grads['gamma' + indice] = dgamma
            grads['beta' + indice]  = dbeta
            
            dout_next = dout_cur
            
          
        return loss, grads