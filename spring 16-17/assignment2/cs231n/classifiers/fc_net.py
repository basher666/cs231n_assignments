from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        D=np.prod(input_dim)
        H=hidden_dim
        C=num_classes
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ##########################################################################
        
        W1=np.random.normal(scale=weight_scale,size=(D,H))
        b1=np.zeros(H)
        W2=np.random.normal(scale=weight_scale,size=(H,C))
        b2=np.zeros(C)
        self.params['W1']=W1
        self.params['b1']=b1
        self.params['W2']=W2
        self.params['b2']=b2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']
        
        out_aff_1,cache_aff_1=affine_forward(X,W1,b1)

        relu_out ,relu_cache=relu_forward(out_aff_1)

        out_aff_2,cache_aff_2=affine_forward(relu_out,W2,b2)        
        
        scores=out_aff_2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss,grad_softmax=softmax_loss(out_aff_2,y)

        reg_loss=self.reg*np.sum(W1*W1)+self.reg*np.sum(W2*W2)
        reg_loss*=0.5
        loss+=reg_loss

        da1,dw2,db2=affine_backward(grad_softmax,cache_aff_2)
        grads['W2']=dw2+self.reg*W2
        grads['b2']=db2
        
        dz1=relu_backward(da1,relu_cache)

        dx1,dw1,db1=affine_backward(dz1,cache_aff_1)
        grads['W1']=dw1+self.reg*W1
        grads['b1']=db1
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
    
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        H=hidden_dims
        L=self.num_layers
        D=input_dim
        C=num_classes
        
        '''
        self.params['W1']=np.random.normal(scale=weight_scale,size=(D,H[0]))
        self.params['b1']=np.zeros(H[0])
        self.params['W2']=np.random.normal(scale=weight_scale,size=(H[0],H[1]))
        self.params['b2']=np.zeros(H[1])
        
        self.params['W3']=np.random.normal(scale=weight_scale,size=(H[1],H[2]))
        self.params['b3']=np.zeros(H[2])
        '''

        string_w='W'
        string_b='b'

        for i in range(L):
            if i==0:
                self.params[string_w+`i+1`]=np.random.normal(scale=weight_scale,size=(D,H[i]))
                self.params[string_b+`i+1`]=np.zeros(H[i])

            elif i==L-1:
                self.params[string_w+`i+1`]=np.random.normal(scale=weight_scale,size=(H[i-1],C))
                self.params[string_b+`i+1`]=np.zeros(C)

            else:
                self.params[string_w+`i+1`]=np.random.normal(scale=weight_scale,size=(H[i-1],H[i]))
                self.params[string_b+`i+1`]=np.zeros(H[i])

            if self.use_batchnorm and i<L-1:
                self.params['gamma'+`i+1`]=np.ones(H[i])
                self.params['beta'+`i+1`]=np.zeros(H[i])
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        L=self.num_layers

        out_aff_i=[]
        cache_aff_i=[]
        bn_out_i=[]
        bn_cache_i=[]
        relu_out_i=[]
        relu_cache_i=[]
        dropout_cache_i=[]
        dropout_out_i=[]
        string_w='W'
        string_b='b'
        
        out_aff_tmp , cache_aff_tmp,relu_out_tmp,relu_cache_tmp, bn_out_tmp, bn_cache_tmp ,dropout_out_tmp,dropout_cache_tmp = None, None, None, None ,None ,None ,None,None
        
        for i in range(L-1):
            if i==0:
                
                out_aff_tmp,cache_aff_tmp=affine_forward(X,self.params[string_w+`i+1`],self.params[string_b+`i+1`])
                
            else:
                if self.use_dropout:
                    out_aff_tmp, cache_aff_tmp = affine_forward(dropout_out_i[i-1],self.params[string_w+`i+1`],self.params[string_b+`i+1`])
                else:
                    out_aff_tmp,cache_aff_tmp=affine_forward(relu_out_i[i-1],self.params[string_w+`i+1`],self.params[string_b+`i+1`])

            if self.use_batchnorm :
                if self.use_dropout:
                    bn_out_tmp, bn_cache_tmp = batchnorm_forward(out_aff_tmp, self.params['gamma'+`i+1`], self.params['beta'+`i+1`], self.bn_params[i])
                    relu_out_tmp,relu_cache_tmp=relu_forward(bn_out_tmp)
                    dropout_out_tmp , dropout_cache_tmp=dropout_forward(relu_out_tmp,self.dropout_param)
                    bn_out_i.append(bn_out_tmp)
                    bn_cache_i.append(bn_cache_tmp)
                    dropout_cache_i.append(dropout_cache_tmp)
                    dropout_out_i.append(dropout_out_tmp)
                else:
                    bn_out_tmp, bn_cache_tmp = batchnorm_forward(out_aff_tmp, self.params['gamma'+`i+1`], self.params['beta'+`i+1`], self.bn_params[i])
                    relu_out_tmp,relu_cache_tmp=relu_forward(bn_out_tmp)
                    bn_out_i.append(bn_out_tmp)
                    bn_cache_i.append(bn_cache_tmp)
          
          
            else:
                if self.use_dropout:                
                    relu_out_tmp, relu_cache_tmp = relu_forward(out_aff_tmp)
                    dropout_out_tmp , dropout_cache_tmp=dropout_forward(relu_out_tmp,self.dropout_param)
                    dropout_cache_i.append(dropout_cache_tmp)
                    dropout_out_i.append(dropout_out_tmp)
                else:
                    relu_out_tmp, relu_cache_tmp = relu_forward(out_aff_tmp)
                    
                
            relu_out_i.append(relu_out_tmp)
            relu_cache_i.append(relu_cache_tmp)
            out_aff_i.append(out_aff_tmp)
            cache_aff_i.append(cache_aff_tmp)

        out_aff_tmp , cache_aff_tmp=affine_forward(relu_out_i[L-2],self.params[string_w+`L`],self.params[string_b+`L`])

        
        out_aff_i.append(out_aff_tmp)
        cache_aff_i.append(cache_aff_tmp)
        
        scores=out_aff_i[L-1]
        C=scores.shape[1]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss,grads_softmax = softmax_loss(out_aff_i[L-1],y)
        reg_loss=0.0
        for i in range(L):
            reg_loss+=(self.reg*np.sum(self.params[string_w+`i+1`]*self.params[string_w+`i+1`]))

        reg_loss*=0.5
        loss+=reg_loss
        dz=grads_softmax
        #print('dz_shape=',dz.shape)
        for i in range(L-1,0,-1):
            da,grads[string_w+`i+1`],grads[string_b+`i+1`]=affine_backward(dz,cache_aff_i[i])
            if self.use_dropout:
                dd = dropout_backward(da,dropout_cache_i[i-1])
                dz = relu_backward(dd,relu_cache_i[i-1])
            else:
                dz = relu_backward(da,relu_cache_i[i-1])
            if self.use_batchnorm:
                dz, grads['gamma'+`i`] , grads['beta'+`i`] =batchnorm_backward(dz,bn_cache_i[i-1])
            #print('dz_shape=',dz.shape)
        
        #print('dz_shape=',dz.shape)
        
        dx,grads[string_w+'1'],grads[string_b+'1']=affine_backward(dz,cache_aff_i[0])

        for i in range(0,L):
            grads[string_w+`i+1`]+=self.reg*self.params[string_w+`i+1`]

        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
