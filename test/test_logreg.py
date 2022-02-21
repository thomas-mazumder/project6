from regression.logreg import LogisticRegression
from regression.utils import loadDataset
import numpy as np

"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

def test_calculate_gradient():
    """
    Check that gradient is computed correctly
    Numbers are from https://web.stanford.edu/~jurafsky/slp3/5.pdf page 17
    """
    lr = LogisticRegression(num_feats=2)
    lr.W = np.array([0,0,0])
    X = np.array([[3, 2, 1],
                  [3, 2, 1]])
    y = np.array([1, 1])
    grad = lr.calculate_gradient(X, y) 
    assert np.all(grad == np.array([-1.5, -1.0, -.5]))        

def test_weights_update():
    """
    Check that the weights update after one iteration of gradient calculation
    Numbers are as in test_calculate_gradient()
    """
    lr = LogisticRegression(num_feats=2, learning_rate=0.1, max_iter = 2)
    lr.W = np.array([0,0,0])
    X = np.array([[3, 2],
                  [3, 2]])
    y = np.array([1, 1])
    lr.train_model(X, y, X, y)
    assert np.allclose(lr.W, np.array([0.15, 0.1, 0.05]))

def test_loss_decreases():
    """
    Check that the loss history of the model decreases steadily
    """
    X_train, X_val, y_train, y_val = loadDataset(split_percent = .8)
    lr = LogisticRegression(6, batch_size = 12, learning_rate=0.005)
    np.random.seed(0)
    lr.W = np.random.randn(lr.num_feats + 1).flatten()
    lr.train_model(X_train, y_train, X_val, y_val)	
    # check that the highest loss is seen early in training, 
    # which is expected to happen unless the initialization of W was "lucky"
    assert np.argmax(lr.loss_history_train) < 10
    # check whether the loss ends at a low value
    assert lr.loss_history_train[-1] < 1.0

def test_accuracy():
    """
    Check thtat the accuracy of the learned model is not terrible on the validation data
    """
    X_train, X_val, y_train, y_val = loadDataset(split_percent = .8)
    lr = LogisticRegression(6, batch_size = 12, learning_rate=0.005)
    np.random.seed(0)
    lr.W = np.random.randn(lr.num_feats + 1).flatten()
    lr.train_model(X_train, y_train, X_val, y_val)
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    y_pred = lr.make_prediction(X_val)
    y_pred = np.round(y_pred)
    acc = np.sum(y_val == y_pred)/len(y_val)
    assert acc > .70
         


