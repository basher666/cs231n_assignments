import numpy as np
from random import shuffle
from past.builtins import xrange
import math

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
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]


  for i in xrange(N):
    scores=np.zeros(D)
    scores=X[i].dot(W)
    scores-=np.max(scores)
    loss-=scores[y[i]]
    normalization_const=np.sum(np.exp(scores))
    loss+=math.log(normalization_const)
    for j in xrange(C):
      expo=math.exp(scores[j])
      if(j==y[i]):
        dW[:,j]=dW[:,j]+((expo/normalization_const) -1)*X[i]
      else:
        dW[:,j]=dW[:,j]+(expo/normalization_const)*X[i]
     
      
    
    
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################




  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


  
  loss /=N
  loss+= reg * np.sum(W*W)
  dW /=N
  dW += 2 * reg * W

  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]

  for i in xrange(N):
    scores=X[i].dot(W)
    scores-=np.max(scores)
    loss-=scores[y[i]]
    normalization_const=np.sum(np.exp(scores))
    loss+=math.log(normalization_const)
    grad=np.exp(scores)/normalization_const
    grad[y[i]]-= 1
    del_W=np.outer(X[i],grad)
    dW+=del_W
  
  loss /=N
  loss+= reg * np.sum(W*W)
  dW /=N
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

