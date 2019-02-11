# SRCNN, Image Super-Resolution Using Deep Convolutional Networks in Tensorflow.

My implementation of Image Super-Resolution Using Deep Convolutional Networks in Tensorflow.

**The algorithm is not mine**, its an implementation of the SRCNN introduced by [Dong et al](https://arxiv.org/pdf/1501.00092.pdf)

I have supplied a few functions to visulise the weights of the model (kernels) and the feature maps.

**Project Structure:**

- model - you can find the preon trained model, model is trained on a single colour channel, and hence expects this as as in input
- srcnn.py - here I initialise the network, and test it on image supplied in "image" directory
- outputs - self explanatory, you check the results obtained in here

The project is using Python 3.5.

**Network**

- Convolitional Neural Network

- Filter Sizes: 9 - 1 - 5, as in the original paper

- Learning rate is 10^−4 for the first two layers, and 10^−5 for the last layer, according to the authors' benchmark

## List of Dependencies:

- MatPlotLib - for visualisation

- NumPy - for everything, its NumPy

- Tensorflow - network built

- Scipy - for image processing

- Scikit-image - computing PSNR


**NOT JUPYTER NOTEBOOK**

Run from terminal, PyCharm, or from anything else that makes you happy:)

Everything else you need to know is in the comments.

Star, fork, do your thing.
