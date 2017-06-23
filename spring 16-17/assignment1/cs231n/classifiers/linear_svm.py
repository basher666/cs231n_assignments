import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    correct_class_grad=0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      incorrect_class_grad=0
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        incorrect_class_grad=X[i]
        correct_class_grad-=1
      dW[:,j]=dW[:,j] +incorrect_class_grad
      
    correct_class_grad=correct_class_grad*X[i]
    dW[:,y[i]]=dW[:,y[i]]+correct_class_grad



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW=dW+2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = np.float128(0.0)
  dW = np.zeros(W.shape) # initialize the gradient as zero
  temp_ones=np.ones(W.shape[1])
  for i in xrange(X.shape[0]):
    scores=W.T.dot(X[i])
    margins=np.maximum(0,scores-scores[y[i]]+1)
    margins[y[i]]=0
    loss+=np.sum(margins)
    for j in xrange(W.shape[1]):
      if(j==y[i]):
        continue
      if(margins[j]>0):
        dW[:,y[i]]-=X[i]
        dW[:,j]+=X[i]
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  loss /=X.shape[0]
  dW /=X.shape[0]
  loss+=0.5*reg*np.sum(W*W)
  dW+=2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  
  return loss, dW
