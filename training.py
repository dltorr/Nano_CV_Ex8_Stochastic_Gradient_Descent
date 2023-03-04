import argparse
import logging

import tensorflow as tf

from dataset import get_datasets
from logistic import softmax, cross_entropy, accuracy


def get_module_logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def sgd(params, grads, lr, bs):
    """
    stochastic gradient descent implementation
    args:
    - params [list[tensor]]: model params
    - grads [list[tensor]]: param gradient such that params[0].shape == grad[0].shape
    - lr [float]: learning rate
    - bs [int]: batch_size
    """
    # IMPLEMENT THIS FUNCTION
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / bs)

def model(X):
    """
    logistic regression model
    """
    flatten_X = tf.reshape(X, (-1, W.shape[0]))
    return softmax(tf.matmul(flatten_X, W) + b)



def training_loop(lr):
    """
    training loop
    args:
    - lr [float]: learning rate
    returns:
    - mean_acc [tensor]: training accuracy
    - mean_loss [tensor]: training loss
    """
    accuracies = []
    losses = []
    for X, Y in train_dataset:
        '''
        print('All X')
        print(X)
        print('All Y')
        print(Y)
        print('One X')
        print(X[0])
        print('One Y')
        print(Y[0])
        
        print('Model with one input') 
        print(model(X[0]))
        print('Model with all inputs')
        print(model(X))
        '''
        with tf.GradientTape() as tape:
            # IMPLEMENT THIS FUNCTION
            # 1. Input normalisation
            X = X / 255.0
            # Watch the input to the model
            #tape.watch(X)
            # 2. Run the model
            Y_hat = model(X)
            # 3. Calculate the loss
            # 3.1. Using one_hot vector
            one_hot = tf.one_hot(Y, 43)
            # 3.2. Calculate the loss with the cross entropy
            loss = cross_entropy(Y_hat, one_hot)
            losses.append(tf.math.reduce_mean(loss))
            # Calculate the gradient
            grads = tape.gradient(loss, [W, b])
            # stochastic gradient descent
            sgd([W, b], grads, lr, X.shape[0]) 
            # Calculate accuracy 
            acc = accuracy(Y_hat, Y)
            accuracies.append(acc)
        
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    mean_loss = tf.math.reduce_mean(losses)
    return mean_loss, mean_acc


def validation_loop(val_dataset, model):
    """
    validation loop
    args:
    - val_dataset: 
    - model [func]: model function
    returns:
    - mean_acc [tensor]: mean validation accuracy
    """
    # IMPLEMENT THIS FUNCTION
    accuracies = []
    losses = []
    for X, Y in train_dataset:
        with tf.GradientTape() as tape:
            # IMPLEMENT THIS FUNCTION
            # 1. Input normalisation
            X = X / 255.0
            # Watch the input to the model
            #tape.watch(X)
            # 2. Run the model
            Y_hat = model(X)
            # 3. Calculate the loss
            # 3.1. Using one_hot vector
            one_hot = tf.one_hot(Y, 43)
            # 3.2. Calculate the loss with the cross entropy
            loss = cross_entropy(Y_hat, one_hot)
            losses.append(tf.math.reduce_mean(loss))
            # Calculate the gradient
            grads = tape.gradient(loss, [W, b])
            # stochastic gradient descent
            #sgd([W, b], grads, lr, X.shape[0]) 
            # Calculate accuracy 
            acc = accuracy(Y_hat, Y)
            accuracies.append(acc)
        
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    #mean_loss = tf.math.reduce_mean(losses)
    return mean_acc


if __name__  == '__main__':
    logger = get_module_logger(__name__)
    # Inputs to the function when calling it
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--imdir', required=True, type=str,
                        help='data directory')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()    

    logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    # get the datasets
    # Get the data from the specified input
    train_dataset, val_dataset = get_datasets(args.imdir)
    
    # each dataset with the following structure : imdir, image_size, batch_size=256, validation_split, subset='training', seed
    # set the variables
    
    num_inputs = 1024*3
    num_outputs = 43
    # Initialize with a random values with the proper shape based on inputs and outputs
    W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                    mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))
    # Learning rate of 0.1
    lr = 0.1
    print('logger')
    print(logger)
    print('parser')
    print(parser)
    print('args')
    print(args)
    print('args.imdir')
    print(args.imdir)
    # training! 
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch}')
        loss, acc = training_loop(lr)
        logger.info(f'Mean training loss: {loss}, mean training accuracy {acc}')
        acc = validation_loop(val_dataset, model)
        logger.info(f'Mean validation accuracy {acc}')
    