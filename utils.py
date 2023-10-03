import numpy as np

def normalize(x_train, x_test):
  mean = x_train.mean(axis=(0,1))
  std  = x_train.std(axis=(0,1))
  return (x_train-mean)/(std+np.finfo(float).eps), (x_test-mean)/(std+np.finfo(float).eps)

def onehot(y):
  n = np.size(y)
  y_onehot = np.zeros((n,2))
  y_onehot[y ==-1, 0] = 1
  y_onehot[y == 1, 1] = 1
  
  return y_onehot