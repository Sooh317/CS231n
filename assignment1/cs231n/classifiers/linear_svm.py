from builtins import range
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                ### added by me ###
                dW[:,j] += X[i]
                dW[:, y[i]] -= X[i]
                    ### end ###

    ### added by me ###
    dW /= num_train      # 'cause we take the average of all loses
    dW += 2*reg*W        # don't forget the regularization term
       ### end ###


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # f(x) = Wx
    # L_i = \sigma_{j != y_i} max(0, W[j, :].dot(x_i) - W[y_i,:].dot(x_i) + 1)
    # \nabla_{W[y_i, :]} L_i = \sigma_{j != y_i} -1 * (1[W[j, :].dot(x_i) - W[y_i,:].dot(x_i) + 1 > 0]) * x_i
    # \nabla_{W[j, :]} L_i = \sigma_{k != y_i} (1[W[j, :].dot(x_i) - W[y_i,:].dot(x_i) + 1 > 0]) * x_i

    # Note that the way this function takes Matrix W is different from what I've just explained

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W) #(N, C)
    tmp = scores[np.arange(num_train), y] # array indexing
    correct_scores = np.reshape(tmp, (num_train, 1)) #(N, 1)
    diff = scores - correct_scores + 1 # using broadcast
    margins = np.maximum(0, diff)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """ my first code 

    exist = (margins > 0).astype(np.int) # it has some value relating to the gradient if margin > 0
    dW += X.T.dot(exist) # compute it here except for y[i]

    # make a matrix where only (i, y[i]) contains a value describing how many times X[i] should be subtracted
    mat = np.zeros((num_train, num_classes))
    mat[np.arange(num_train), y] = np.sum(exist, axis=1)
    dW -= X.T.dot(mat)
    dW /= num_train

    # regularization term
    dW += 2*reg*W

    """

    margins[margins > 0] = 1
    margins[np.arange(num_train), y] = -1 * np.sum(margins, axis=1)
    dW += X.T.dot(margins)
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
