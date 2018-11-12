import tensorflow as tf

class CosineComparison(tf.keras.layers.Layer):
     def __init__(self):
         super(CosineComparison, self).__init__()

     ''' Returns a matrix of cosine-distance comparisons of all vectors in X.

         Params:
                X: n x d matrix (float32)
         Returns:
                cosine_distance: a n x n matrix where
                                 cosine_distance[i, j] = cosine-distance(X[i, :], X[j, :])'''
     def call(self, X):
         dot_product_X = tf.to_float(tf.matmul(a = X, b = X, transpose_b = True))
         diag = tf.reshape(tf.sqrt(tf.to_float(tf.matrix_diag_part(dot_product_X))), [1,-1]) # 1 x n
         norm_products = tf.matmul(a = diag, b = diag, transpose_a = True)
         cosine_distance = tf.divide(x = dot_product_X, y = norm_products)
         return cosine_distance

# How are we training
# - how do we pick from here
#   - loss against matrix (like an image mse) --> but how do we know true values (just 1 for dup
#                                                                                 0 elsewhere)

# Are we taking the absolute value?
