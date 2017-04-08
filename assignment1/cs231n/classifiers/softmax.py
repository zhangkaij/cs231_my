import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_class = W.shape[1]
  num_train = X.shape[0]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        
        sum_score = 0
        for score in scores:
            sum_score += np.exp(score)
            
        #此处犯了一个错误刚开始使用的np.log10(exp_loss)，导致计算结果不对
        loss += np.log(sum_score)
        loss -= scores[y[i]]                 
        
        for j in range(num_class):
            p = np.exp(scores[j])/sum_score
            dW[:, j] += (p - (j == y[i])) * X[i]
            
  loss /= num_train  
  loss += 0.5 * reg * np.sum(W * W)
  #dW不要忘记除以num_train
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #print(dW.shape)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  loss_vec = -scores[range(num_train), y] + np.log(np.sum(np.exp(scores), axis = 1))
  loss = np.sum(loss_vec) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  exp_scores = np.exp(scores)
  p = exp_scores / np.reshape(np.sum(exp_scores, axis = 1), (num_train, 1))
  p[range(num_train), y] += -1
  dW = (X.T).dot(p) / num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

