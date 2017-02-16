# https://www.tensorflow.org/get_started/get_started

import tensorflow as tf
import numpy as np
import pandas as pd

def constants():
    constant3, constant4 = _get_constants()
    session = tf.Session()
    print(constant3, constant4)
    print(session.run([constant3, constant4]))

def add_nodes():
    add_node = _get_add_node(*_get_constants())
    session = tf.Session()
    print(add_node)
    print(session.run(add_node))

def placeholders():
    placeholder_node1 = tf.placeholder(tf.float32)
    placeholder_node2 = tf.placeholder(tf.float32)
    add_node = placeholder_node1 + placeholder_node2
    session = tf.Session()
    print(session.run(add_node, {placeholder_node1: 3, placeholder_node2: 4.5}))
    print(session.run(add_node, {placeholder_node1: [1,2], placeholder_node2: [5,7]}))
    add_and_triple = add_node * 3
    print(session.run(add_and_triple, {placeholder_node1: 3, placeholder_node2: 4.5}))

def basic_linear_model():
    W, b, x, linear_model = _get_linear_model()
    session = _init_and_get_session()
    # Prints Ys
    print(session.run(linear_model, {x:[1,2,3]}))

    # Calculate loss
    y, loss = _get_loss(linear_model)
    print(session.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # 23.66

    # Update W and b so that the model fits the data perfectly
    fixW = tf.assign(W, [-1])
    fixb = tf.assign(b, [1])
    session.run([fixW, fixb])
    print(session.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # 0

def basic_gradient_descent():
    W, b, x, linear_model = _get_linear_model()
    y, loss = _get_loss(linear_model)
    session = _init_and_get_session()

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    data = { x: [1,2,3,4],
             y: [0,-1,-2,-3]}

    print("\nModel pre-optimize loss (expect 23.66): {0:.2f}".format(session.run(loss, data)))
    print("Model pre-optimize [W,b]: {}".format(session.run([W,b])))

    for i in range(1000):
        session.run(train, data)

    print("Model post-optimize loss (expect 0.00): {0:.2f}".format(session.run(loss, data)))
    print("Model post-optimize [W,b]: {}".format(session.run([W,b])))


def linear_regression_again():
    # TODO - broken
    features = [tf.contrib.layers.real_valued_column("",dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    data = tf.contrib.learn.datasets.base.Dataset(
            data = tf.constant(np.array([[1],[2],[3],[4]])),
            target = tf.constant(np.array([[0],[-1],[-2],[-3]])))

    def input_fn():
        #return data.data, data.target
        return tf.contrib.layers.real_valued_column(column_name='x', default_value=[[1],[2],[3],[4]]), tf.contrib.layers.real_valued_column(column_name='y', default_value=[[0],[-1],[-2],[-3]])
        #return tf.constant(np.array([[1],[2],[3],[4]])), tf.constant(np.array([[0],[-1],[-2],[-3]]))
        #return tf.constant([[1,2,3,4]]), tf.constant([0,-1,-2,-3])
        #return tf.contrib.learn.datasets.base.Dataset(
        #    data = np.array([[1],[2],[3],[4]]),
        #    target = np.array([[0],[-1],[-2],[-3]]))
        #return np.array([[1],[2],[3],[4]]), np.array([[0],[-1],[-2],[-3]])

    estimator.fit(input_fn=input_fn, steps=1000)
    print(estimator.predict(x=2))
    #estimator.fit(x=data.data, y=data.target, steps=1000)
    #estimator.evaluate(input_fn=input_fn)

def _get_constants():
    return tf.constant(3.0, tf.float32), tf.constant(4.0)

def _get_add_node(node1, node2):
    return tf.add(node1, node2)

def _get_linear_model():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W*x+b
    return W, b, x, linear_model

def _get_loss(linear_model):
    y = tf.placeholder(tf.float32)
    squared_diff = tf.square(linear_model-y)
    return y, tf.reduce_sum(squared_diff)

def _init_and_get_session():
    session = tf.Session()
    # Variables aren't initialized until this is called
    initialize = tf.global_variables_initializer()
    session.run(initialize)
    return session
 

if __name__ == "__main__":
    #constants()
    #add_nodes()
    #placeholders()
    #basic_linear_model()
    #basic_gradient_descent()
    linear_regression_again()
