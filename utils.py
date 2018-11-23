# utils.py
# Zhao Shen, Arun Pidugu, Arnav Ghosh
# 19th Nov. 2018

from keras import backend as K

# from keras mnist tutorials
def euclidean_distance(vect_lst):
    x, y = vect_lst
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# from keras mnist tutorials
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Hadsell-et-al.'06: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
# From keras mnist
# 0 is more similar
def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
