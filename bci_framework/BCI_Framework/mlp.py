"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
# __docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression, load_data
# import matplotlib.pyplot as plt


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, learning_rate, L1_reg, L2_reg, n_hidden, best_error = None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.best_error = best_error
        self.n_epochs = 80
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.n_hidden = n_hidden

    def fit(self, X, Y):
        """
        Demonstrate stochastic gradient descent optimization for a multilayer
        perceptron
      
        This is demonstrated on MNIST.
      
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient
      
        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)
      
        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)
      
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
      
        :type dataset: string
        :param dataset: the path of the MNIST dataset file from
                     http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
      
      
       """
        
        
        if (self.best_error is  None):
            train_set_x, train_set_y = shared_dataset((X[0], Y[0]-1), borrow=True)
            valid_set_x, valid_set_y = shared_dataset((X[1], Y[1]-1), borrow=True)
            n_classes = len(set(Y[0]))
            input_dimension = X[0].shape[1]
            validation_size = Y[1].shape[0]
            train_size = X[0].shape[0]

        else:
            Y = numpy.array(Y)
            train_set_x, train_set_y = shared_dataset((X, Y-1), borrow=True)
            n_classes = len(set(Y))
            input_dimension = X.shape[1]
            train_size = X.shape[0]
        
      
        n_batches = 1
      
        ######################
        # BUILD ACTUAL MODEL #
        ######################
#         print '... building the model'
      
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
      
        rng = numpy.random.RandomState(1234)
      
        # construct the MLP class
#         classifier = MLP(rng=rng, input=x, n_in=datasets[0].shape[1],
#                          n_hidden=n_hidden, n_out=n_classes)
#         
        # Since we are dealing with a one hidden layer MLP, this will
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer
        # implementing any other nonlinearity
        self.hiddenLayer = HiddenLayer(rng=rng, input=self.x,
                                       n_in= input_dimension, n_out=self.n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=self.n_hidden,
            n_out=n_classes)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
        self.probabilities = self.logRegressionLayer.p_y_given_x

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = self.negative_log_likelihood(self.y) \
             + self.L1_reg * self.L1 \
             + self.L2_reg * self.L2_sqr
      
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
    #     test_model = theano.function(inputs=[index],
    #             outputs=classifier.errors(y),
    #             givens={
    #                 x: test_set_x[index * batch_size:(index + 1) * batch_size],
    #                 y: test_set_y[index * batch_size:(index + 1) * batch_size]})
      
        
        if (self.best_error is  None):
            self.validate_model = theano.function(inputs=[],
                    outputs=(self.errors(self.y),self.probabilities),
                    givens={
                        self.x: valid_set_x,
                        self.y: valid_set_y})
      
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
      
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = []
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
      
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(inputs=[], outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x,
                    self.y: train_set_y})
      
        ###############
        # TRAIN MODEL #
        ###############
#         print '... training'
      
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = 1
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
      
        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
      
        epoch = 0
        done_looping = False
        
        errors_for_plot = numpy.zeros(self.n_epochs)
        
        
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
    #         for minibatch_index in xrange(n_train_batches):
    #         minibatch_index = 0
            minibatch_avg_cost = self.train_model()
#             print 'trainig error: ', minibatch_avg_cost
            if not (self.best_error is None) and minibatch_avg_cost <= self.best_error:#this is for test phase
#                 validation_losses, my_probs = self.validate_model()
                break
            elif self.best_error is None:
                # iteration number
                iter = (epoch - 1) * n_batches
         
                if (iter + 1) % validation_frequency == 0:
                   # compute zero-one loss on validation set
                   validation_losses, my_probs = self.validate_model()
                     
                   this_validation_loss = numpy.mean(validation_losses)
                   errors_for_plot[epoch-1] = this_validation_loss
                   
#                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
#                         (epoch, 1, n_batches,
#                          this_validation_loss * 100.))
#          
                   # if we got the best validation score until now
                   if this_validation_loss < best_validation_loss:
                       #improve patience if loss improvement is good enough
                       if this_validation_loss < best_validation_loss *  \
                              improvement_threshold:
                           patience = max(patience, iter * patience_increase)
         
                       best_validation_loss = this_validation_loss
                       best_iter = iter
         
                       # test it on the test set
        #                     test_losses = [test_model(i) for i
        #                                    in xrange(n_test_batches)]
        #                     test_score = numpy.mean(test_losses)
         
        #                print(('     epoch %i, minibatch %i/%i, test error of '
        #                       'best model %f %%') %
        #                      (epoch, minibatch_index + 1, n_train_batches,
        #                       test_score * 100.))
         
                if patience <= iter:
                       done_looping = True
                       break
          
        end_time = time.clock()

        
#         print(('Optimization complete. Best validation score of %f %% '
#                'obtained at iteration %i') %
#               (best_validation_loss * 100., best_iter + 1))
#          
#         print >> sys.stderr, ('The code for file ' +
#                               os.path.split(__file__)[1] +
#                               ' ran for %.2fm' % ((end_time - start_time) / 60.))
#         
#         plt.plot(numpy.arange(100), errors_for_plot)
#         plt.show()

        return best_validation_loss

    def predict(self, X):
        

        test_set_x, test_set_y = shared_dataset((X, numpy.array([])), borrow=True)  
      
        test_model = theano.function(inputs=[],
        outputs=self.logRegressionLayer.y_pred,
            givens={self.x: test_set_x})
#       
        labels = test_model() + 1
        return labels
    
    def predict_proba(self, X):
        
        test_set_x, test_set_y = shared_dataset((X, numpy.array([])), borrow=True)  
      
        test_model = theano.function(inputs=[],
        outputs=self.logRegressionLayer.p_y_given_x,
            givens={self.x: test_set_x})
#         self.validate_model = theano.function(inputs=[],
#                 outputs=(self.probabilities),
#                 givens={self.y: valid_set_x} )
#       
        probs = test_model()
        return probs











def apply_mlp(learning_rate, L1_reg, L2_reg, n_hidden, datasets):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron
  
    This is demonstrated on MNIST.
  
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient
  
    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)
  
    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)
  
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
  
    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
  
  
   """
    
    datasets1 = load_data('mnist.pkl.gz')
#  
    train_set_x1, train_set_y1 = datasets1[0]
    valid_set_x1, valid_set_y1 = datasets1[1]
    test_set_x1, test_set_y1 = datasets1[2]
#  
    
    n_epochs = 100
#     train_set_x = datasets[0]
#     train_set_y = datasets[1]
#     valid_set_x = datasets[2]
#     valid_set_y = datasets[3]
    train_set_x, train_set_y = shared_dataset((datasets[0],datasets[1]-1), borrow=True)
    valid_set_x, valid_set_y = shared_dataset((datasets[2],datasets[3]-1), borrow=True)  
#     test_set_x = datasets[4]
#     test_set_y = datasets[5]
  
    n_classes = len(set(datasets[1]))
    # compute number of minibatches for training, validation and testing
    n_train_batches = 1
    n_valid_batches = 1
    n_test_batches = 1
  
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
  
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  #self.y the labels are presented as 1D vector of
                        # [int] labels
  
    rng = numpy.random.RandomState(1234)
  
    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=datasets[0].shape[1],
                     n_hidden=n_hidden, n_out=n_classes)
    
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr
  
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
#     test_model = theano.function(inputs=[index],
#             outputs=classifier.errors(y),
#             givens={
#                 x: test_set_x[index * batch_size:(index + 1) * batch_size],
#                 y: test_set_y[index * batch_size:(index + 1) * batch_size]})
  
    validation_size = datasets[3].shape[0]
    train_size = datasets[0].shape[0]
      
    validate_model = theano.function(inputs=[],
            outputs=(classifier.errors(y),classifier.probabilities),
            givens={
                x: valid_set_x,
                y: valid_set_y})
  
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
  
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))
  
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x,
                y: train_set_y})
  
    ###############
    # TRAIN MODEL #
    ###############
#     print '... training'
  
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 1
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
  
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
  
    epoch = 0
    done_looping = False
  
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
#         for minibatch_index in xrange(n_train_batches):
#         minibatch_index = 0
        minibatch_avg_cost = train_model()
#         print 'trainig error: ', minibatch_avg_cost
        # iteration number
        iter = (epoch - 1) * n_train_batches
 
        if (iter + 1) % validation_frequency == 0:
           # compute zero-one loss on validation set
           validation_losses, my_probs = validate_model()
             
           this_validation_loss = numpy.mean(validation_losses)
 
#            print('epoch %i, minibatch %i/%i, validation error %f %%' %
#                 (epoch, 1, n_train_batches,
#                  this_validation_loss * 100.))
 
           # if we got the best validation score until now
           if this_validation_loss < best_validation_loss:
               #improve patience if loss improvement is good enough
               if this_validation_loss < best_validation_loss *  \
                      improvement_threshold:
                   patience = max(patience, iter * patience_increase)
 
               best_validation_loss = this_validation_loss
               best_iter = iter
 
               # test it on the test set
#                     test_losses = [test_model(i) for i
#                                    in xrange(n_test_batches)]
#                     test_score = numpy.mean(test_losses)
 
#                print(('     epoch %i, minibatch %i/%i, test error of '
#                       'best model %f %%') %
#                      (epoch, minibatch_index + 1, n_train_batches,
#                       test_score * 100.))
 
        if patience <= iter:
               done_looping = True
               break
  
    end_time = time.clock()
#     print(('Optimization complete. Best validation score of %f %% '
#            'obtained at iteration %i, with test performance %f %%') %
#           (best_validation_loss * 100., best_iter + 1, test_score * 100.))
#     
#     print(('Optimization complete. Best validation score of %f %% '
#            'obtained at iteration %i') %
#           (best_validation_loss * 100., best_iter + 1))
#      
#     print >> sys.stderr, ('The code for file ' +
#                           os.path.split(__file__)[1] +
#                           ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    return best_validation_loss

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return (shared_x, T.cast(shared_y, 'int32'))


if __name__ == '__main__':
    test_mlp()