# utils.py
# Zhao Shen, Arun Pidugu, Arnav Ghosh
# 19th Nov. 2018

from keras import backend as K

def euclidean_distance(vect_lst):
    x, y = vect_lst
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    pass

# Will have to modify what we mean by accuracy then
