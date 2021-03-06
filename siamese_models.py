# siamese_models.py
# Zhao Shen, Arun Pidugu, Arnav Ghosh
# 19th Nov. 2018

from keras import backend as K
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Lambda, Input
from keras.models import Sequential, Model
import numpy as np
import tensorflow as tf

import utils

# ============== CONSTANTS ============== #
ENCODED_SIZE = 4096

# ============== SINGLE SIAMESE STRUCTURES ============== #

# Two branch siamese based on http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
# with changes to kernel sizes (because our input shape is 7 x 7) and removal of the final module
def traditional_siamese_branch(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_shape, padding='same', name="siamese_conv1"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', name='siaemese_maxpool1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name="siamese_conv2"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', name='siaemese_maxpool2'))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same', name="siamese_conv3"))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', name='siaemese_maxpool3'))
    #model.add(Conv2D(256, (2, 2), activation='relu', padding='same', name="siamese_conv4"))
    model.add(Flatten(), name='siamese_flatten')
    model.add(Dense(ENCODED_SIZE, activation='sigmoid', name='siamese_branch_dense')) #TODO: RELU or sig.

    return model

# should be able to take a list of inputs and then merge them as necesary
# returns a flat vector repr. of the feature map
# same structure as Omniglot but from https://arxiv.org/pdf/1504.03641.pdf
def two_channel_siamese(input_shape):
    pass
    # literally the same as above --> but with the initialized weights

# ============== COMBINING SIAMESE NETS ============== #

# roi_feature_shape: t x w x h x c where t: #tensors
#                                        w, h, c as necessary
# Returns : models that expects all ROI Features as input (list)
#                       outputs a t*(t - 1) vector of 0, 1s where v[i] indicates whether
#                                                           _______________ are duplicates
def multiple_traditional_siamese_nets(roi_feature_shape):
    t, w, h, c = roi_feature_shape
    inputs = [Input((w, h, c)) for _ in range(t)]
    all_predictions = []

    siamese_model = traditional_siamese_branch((w, h, c)) #to ensure weight sharing
    L1_layer = Lambda(lambda features : K.abs(features[0] - features[1])) #TODO why use L1
    prediction_layer = Dense(1, activation='sigmoid') #awaiting L1 input

    for i in range(t):
        for j in range(i + 1, t):
            left_features = siamese_model(inputs[i])
            right_features = siamese_model(inputs[j])
            L1_distance = L1_layer([left_features, right_features])
            prediction = prediction_layer(L1_distance)
            all_predictions.append(prediction)

    multiple_siamese_model = Model(input=inputs, outputs=all_predictions)
    return multiple_siamese_model

def multiple_distance_siamese_nets(roi_feature_shape):
    t, w, h, c = roi_feature_shape
    inputs = [Input((w, h, c)) for _ in range(t)]
    all_distances = []

    siamese_model = traditional_siamese_branch((w, h, c)) #to ensure weight sharing
    distance = Lambda(utils.euclidean_distance, output_shape=utils.eucl_dist_output_shape)

    for i in range(t):
        for j in range(i + 1, t):
            left_features = siamese_model(inputs[i])
            right_features = siamese_model(inputs[j])
            embed_distance = distance([left_features, right_features]) #TODO check dimension
            all_distances.append(embed_distance)

    multiple_distance_siamese_model = Model(input=inputs, output=all_distances)
    return multiple_distance_siamese_model

# a single roi_feature_shape
def multiple_two_channel_nets(roi_feature_shape):
    t, w, h, c = roi_feature_shape
    inputs = [Input((w, h, c)) for _ in range(t)]
    all_predictions = []

    two_channel_siamese = two_channel_siamese((w, h, c)) #to ensure weight sharing
    prediction_layer = Dense(1, activation='sigmoid') #awaiting flattened input

    for i in range(t):
        for j in range(i + 1, t):
            feature_vector = two_channel_siamese([inputs[i], inputs[j]])
            prediction = prediction_layer(feature_vector)
            all_predictions.append(prediction)

    multiple_two_channel_model = Model(input=inputs, outputs=all_predictions)
    return multiple_two_channel_model

# ============== INITIALIZERS ============== #
# kernel shape --> (w x h x d X k) where d is even
def init_comparison_weights(kernel_shape, dtype=None):
    print(kernel_shape)
    w, h, d, k = kernel_shape #k is the number of kernels
    weights = np.ones((kernel_shape), dtype=dtype)
    weights[:, :, int(d / 2):, :] = -1.0
    return weights

# ============== DEBUGGERS ============== #
def check_variables():
    for v in tf.global_variables():
        print(v.name)

# ==== RESEARCH QUESTIONS ====
# 1. Do the Feature Maps generated by the the adobe paper lend themselves to effective
#    comparison for copy - move detection (or are the feature maps sufficiently general
#    in which case duplicate objects or patches - or patches from similar regions -
#    in the image would always result in the lowest similarity and thus, more needs to
#    be done to distinguish them)
# 2. What sub-network for comparison is the most effective:
#    - Previous work has focussed on distilling the image down to feature vectors and then comparing
#      them in some space. However, would since we're dealing with feature maps here could we use
#      use a two channel network with correctly initialized weights to do this comparison

# ==== DESIGN CHOICES ====
# 1. Take the Siamese net suggested in the Omniglot example (uses CE loss), why? (look at other arch. as well) TODO
#    - Shown that the architecture was effective in learning if two images are the 'same' from classes that didn't have
#      too many examples. --> we figured that this would be an effective thing to use because the feature maps aren't really
#      from any definable class and small differences between them could make them from a diff. 'domain' so we want to do one shot
#      learning. (the analogy is that the feature maps are from different domains and the 'sim' metric is whether they're forged)
#      [What if all of them look very similar in actuality tho !!! --> one thing to explore and that the cosine metric will tell us]
#      [Even though it does feature discrimination first and then generalizes but that its capable of doing so is helpful]
#    - Also uses CE loss (which we debate below)
#
# 2. Why retain the same structure for the two channel siamese model?
# 3. How did we adapt the kernel sizes / network for such small feature maps TODO? --> avoid collapsing the info too fast?
# 4. What are the challenges that come with using siamese nets for feature maps
#    (or is it literally the same problem):
#    - Can't really use data augmentation because changing the feature map could mean we're capturing
#      the semantics of a different object
#    - From the onset, no intuitive reasoning of why two feature maps might be more distant from some
#      other FMs than others (FIND RESOURCES TO INFORM US ON THIS) --> is the problem fund. limiting?
#      --> maybe this is why the two channel one will work better because it has an oppurtunity to
#          actually compare the two feature maps and we'll be able to learn a good way to compare them
#          whereas for the standalone branches, the issue is that they learn independent feature vectors,
#          while this may be effective, it'll be harder to do because it will have to learn more nuanced
#          feature vectors to distinguish same but unforged patches while bringing together forged patches
#
# 5. Why use contrastive loss vs. cross entropy loss? (tentative!!!) TODO
#    (see http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf for what it does)
#    - Against Cont. Loss: We would have to turn this into a binary value and how do we know when to do that
#      (technically we're trying to do binary classification) --> nothing like sigmoid that we could use (more complicated)
#    - Against Cont. Loss: Would it be harder to learn a good mapping because for non-forged patches ... [!!!]
#    - Does contrastive loss make less sense because the features maps aren't really coming from any domain?
#      - There's no real notion of how forged two patches are (they are or aren't) and using a distance
#        metric would explicitly mean that we're saying that if one pair is further from another, then it's less forged
#        (but is that not true of prob. --> less likely to be forged?)
#
# 6. Since we're defining a similarity metric based on on forged images,
#    is it correct to use the ones that just generally look for similarity?
#    - The difference in our case is that we're defining 'similarity' to be how likley is it the images
#      our forged --> so even duplicate patches in the same image (like large patches of grass) should
#      be 'far apart' even if theyre the same (A CHALLENGE)
#    - To solve this challenge, we:
#      - Actually do some more convolutions (explaining why we don't use the FMs as is)
#      - Train the network to learn a good embedding
#      - Choose the loss function as is appropriate (5 should explain why the choice of the loss is imp.)
# 7. Why did we try to train independently? (could try joint training if we have time)
#    - Wanted to see how similar the feature maps were or could be made for copy-move images
#      --> and what the best way to compare them would because that's the first step in ensuring that
#          this part of the network actually influences the rest
#    - Lack of computational resources
#    - Next step would be to see how to combine the loss of this subnetwork to ensure that the
#      performance on other forgeries doesn't drop and the copy-move forgery detection acc. inc.

# ==== OUR EVALUATION METHODS ==== TODO

# ==== EXPERIMENTS ==== TODO
# 1. Comparison b/w all methods
# 2. https://arxiv.org/ftp/arxiv/papers/1604/1604.04573.pdf?
# Our eventual goal is to be able to use this subsystem system to inform the rest of the system,
# ie: almost like a proposal system to say that these are the copy-move detections and so it
# doesn't have to directly identify the correct one but we'd like it to be in the rankings (eg: top 5 ish)
# We also want some way of seeing how high the probabilities are --> [TODO!!!]
# perhaps look at the accuracy measure for RPNs  --> Precision and Recall


# ==== EXISTING WORK ====
# https://arxiv.org/pdf/1808.06323.pdf
# https://arxiv.org/pdf/1802.06515.pdf
# Both of the above use siamese networks to detect images that have been post-processed in
# some way as opposed to for the purpose of copy move detection

# ==== FUTURE DIRECTION ====
# 1. Using Cross-Input Neighbourhood Diff.? (as future direction.)
# 2. Could substitude the decision network as necessary
# 3. Learning a global metric that compares the similarity by relating more feature maps
#    (right not we're not enforcing the fact that only one of them should be 1)
