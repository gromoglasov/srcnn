import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
from skimage import measure

"""
Read image using its path.
Default value is gray-scale, and image is read by YCbCr format as the paper said.
"""


def imread(path, is_grayscale=True):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized
    # image = image / 255.
    label_ = label_ / 255.
    in_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
    in_ = scipy.ndimage.interpolation.zoom(in_, (scale/1.), prefilter=False)

    return in_, label_


"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')


# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

"""Define the model layers with three convolutional layers
"""
# conv1 layer with biases and relu : 64 filters with size 9 x 9. Computes feature maps of input low-resolution images
conv1 = tf.nn.relu(tf.nn.conv2d(inputs, filter=weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + biases['b1'])

# conv2 layer with biases and relu: 32 filters with size 1 x 1, to compute non-linear mapping
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, filter=weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2'])

# conv3 layer with biases and NO relu: 1 filter with size 5 x 5, compute the reconstruction of high-resolution image
conv3 = tf.nn.conv2d(conv2, filter=weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b3']

"""Load the pre-trained model file
"""
model_path='./model/model.npy'
model = np.load(model_path, encoding='latin1').item()


def print_model_weights():
    print(model['w1'])
    print(model['w2'])
    print(model['w3'])


# call functions below for to plot filters
# filter dimension 9*9*1, 64 filters
def vis_w1():
    fig = plt.figure(figsize=(8, 8))
    for i in range(64):
        sub = fig.add_subplot(8, 8, i + 1)
        sub.imshow(model['w1'][:, :, 0, i], cmap='gray')
        plt.axis('off')
    plt.show()


# filter dimension 1*1*64 => each row represents a filter, 32 filters
def vis_w2():
    fig = plt.figure(figsize=(8, 8))
    iterator = 0
    for i in range(32):
        for k in range(64):
            sub = fig.add_subplot(32, 64, iterator + 1)
            sub.imshow(model['w2'][:, :, k, i], cmap='gray')
            plt.axis('off')
            iterator += 1
    plt.show()


# filter dimensions 5*5*32, 1 filter
def vis_w3():
    fig = plt.figure(figsize=(8, 8))
    for i in range(32):
        sub = fig.add_subplot(4, 8, i + 1)
        sub.imshow(model['w3'][:, :, i, 0], cmap='gray')
        plt.axis('off')
    plt.show()


"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
    sess.run(weights[key].assign(model[key]))
    print(key)

for key in biases.keys():
    sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, ground_truth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis=0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output_ = sess.run(conv3, feed_dict={inputs: input_})
# change types
out = output_.astype(np.float64)

# save images
gt = scipy.misc.imsave('ground_truth.jpg', ground_truth_image)
blurred = scipy.misc.imsave('blurred.jpg', blurred_image)
sr_cnn_res = scipy.misc.imsave('result.jpg', out[0, :, :, 0])

# crop the ground truth image and compute the PSNR
out_psnr = measure.compare_psnr(ground_truth_image[6:249, 6:249], out[0, :, :, 0])
bic_psnr = measure.compare_psnr(ground_truth_image, blurred_image)

print('\n')
print('Super Res PSNR: ', out_psnr)
print('Bi-cubic PSNR: ', bic_psnr)
print_model_weights()
vis_w2()
