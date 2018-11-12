import tensorflow as tf

from keras import backend as K
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Lambda, Input
from keras.models import Sequential, Model

def single_siamese_branch(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape, name="siamese_conv1"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', name='siaemese_maxpool1'))
    model.add(Conv2D(128, (7,7), activation='relu', name="siamese_conv2"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', name='siaemese_maxpool2'))
    model.add(Conv2D(128, (4,4), activation='relu', name="siamese_conv3"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', name='siaemese_maxpool3'))
    model.add(Conv2D(256, (4,4), activation='relu', name="siamese_conv4"))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    return model

# roi_feature_shape: t x w x h x c where t: #tensors
#                                        w, h, c as necessary
# Returns : models that expects all ROI Features as input (list)
#                       outputs a t*(t - 1) vector of 0, 1s where v[i] indicates whether
#                                                           _______________ are duplicates
def multiple_siamese_network(roi_feature_shape):
    t, w, h, c = roi_feature_shape
    inputs = [Input((w, h, c)) for _ in range(t)]
    all_predictions = []

    siamese_model = single_siamese_branch((w, h, c))
    L1_layer = Lambda(lambda features : K.abs(features[0] - features[1]))
    prediction_layer = Dense(1,activation='sigmoid') #awaiting L1 input

    for i in range(t):
        for j in range(i + 1, t):
            left_features = siamese_model(inputs[i])
            right_features = siamese_model(inputs[j])
            L1_distance = L1_layer([left_features, right_features])
            prediction = prediction_layer(L1_distance)
            all_predictions.append(prediction)

    multiple_siamese_model = Model(input=inputs, outputs=all_predictions)
    return multiple_siamese_model

# input_shape: t x w x h x c where t: #tensors
#                                  w, h, c as necessary
# def mutliple_siamese_network(ROI_features):
#     t, w, h, c = # ROI_features shape TODO
#     siamese_net = create_siamese_network((w, h, c))
#     models = []
#
#     for i in range(t):
#         for j in range(i + 1, t):
#             models.append()
#
#     #reusing in keras == just instantiating once and passing values multiple times
#
#     # Test - two models
#     with tf.variable_scope('traditional_siamese_nets') as scope:
#         models.append(create_siamese_network((w, h, c)))
#     with tf.variable_scope(scope, reuse = True):
#         models.append(create_siamese_network((w, h, c)))
#
#     # with tf.variable_scope('traditional_siamese_nets') as scope:
#     #     for i in range(t):
#     #         for j in range(i + 1, t):
#     #             models.append(create_siamese_network((w, h, c)))
#     #             scope.reuse_variables()
#
#     return models

def check_variables():
    for v in tf.global_variables():
        print(v.name)

# modularize:
#   - two outputs
#   - one output for two TwoChannel
#   - different ways to compare the two outputs
#   - different ways to use the final vector that is produced


# class TwoChannelSiamese(tf.keras.layers.Layer):
#      def __init__(self):
#          super(TwoChannelSiamese, self).__init__()

# kernel shape --> (w x h x d) where d is even
def init_comparison_weights(kernel_shape):
    weights = np.ones((kernel_shape), dtype= np.float32)
    weights[:, :, int(d / 2):] = -1.0
    return weights
