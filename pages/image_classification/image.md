### Image Classification using GluonCV

Implement a tennis ball detector using a pre-trained image classification network from GluonCV.
We'll step through the pipeline, from loading and transforming an input image, to loading and using
a pre-trained model. Since we're only interested in detecting tennis balls, this is a binary classification problem,
which is slightly different to the multi-class classification setup we've seen so far.

This is a part of programming assignment for [AWS Computer Vision: Getting Started with GluonCV](https://www.coursera.org/learn/aws-computer-vision-gluoncv/home/welcome) course in [Coursera](https://www.coursera.org)

#### 1) Setup

We start with some initial setup: importing packages and setting the path to the data.


```python
import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
```


```python
M3_DATA = Path(os.getenv('DATA_DIR', '../../data'), 'module_3')
M3_IMAGES = Path(M3_DATA, 'images')
M3_MODELS = Path(M3_DATA, 'models')

```

#### 2) Loading an image

Your first task is to implement a function that loads an image from disk given a filepath.

It should return an 8-bit image array, that's in MXNet's NDArray format and in HWC layout (i.e. height, width then channel).


```python
def load_image(filepath):
    image = mx.image.imread(filepath)
    return image
```

#### 3) Transforming an image

Up next, you should transform the image so it can be used as input to the pre-trained network.

Since we're going to use an ImageNet pre-trained network, we need to follow the same steps used for ImageNet pre-training.

See the docstring for more details, but don't forget that GluonCV contains a number of utilities and helper functions to make your life easier! Check out the preset transforms.
Should transform image by:
1. Resizing the shortest dimension to 224. e.g (448, 1792) -> (224, 896).
2. Cropping to a center square of dimension (224, 224).
3. Converting the image from HWC layout to CHW layout.
4. Normalizing the image using ImageNet statistics (i.e. per colour channel mean and variance).
5. Creating a batch of 1 image.

```python
def transform_image(array):
    image = gcv.data.transforms.presets.imagenet.transform_eval(array)
    return image
```

#### 4) Loading a model

With the image loaded and transformed, you now need to load a pre-trained classification model.

Choose a MobileNet 1.0 image classification model that's been pre-trained on ImageNet.

<font color='red'>**CAUTION!**</font>: Although the notebook interface has internet connectivity, the **autograders are not permitted to access the internet**. We have already downloaded the correct models and data for you to use so you don't need access to the internet. However, you do need to specify the correct path to the models when loading a model from the Gluon CV Model Zoo using `get_model` or otherwise. Set the `root` parameter to `M3_MODELS`. As an example, you should have something similar to `gcv.model_zoo.get_model(..., root=M3_MODELS)`. Usually, in the real world, you have internet access, so setting the `root` parameter isn't required (and it's set to `~/.mxnet` by default).


```python
def load_pretrained_classification_network():
    model = gcv.model_zoo.get_model('MobileNet1.0', pretrained=True, root = M3_MODELS)
    return model
```

#### 5) Using a model

Your next task is to pass your transformed image through the network to obtain predicted probabilities for all ImageNet classes.

We'll ignore the requirement of creating just a tennis ball classifier for now.

**Hint #1**: Don't forget that you're typically working with a batch of images, even when you only have one image.

**Hint #2**: Remember that the direct outputs of our network aren't probabilities.


```python
def predict_probabilities(network, data):
    prediction = network(data)
    prediction = prediction[0]
    probability = mx.nd.softmax(prediction)
    return probability
```

#### 6) Finding Class Label

Since we're only interested in tennis ball classification for now, we need a method of finding the probability associated with tennis ball out of the 1000 classes.

You should implement a function that returns the index of a given class label (e.g. `admiral` is index `321`)

**Hint**: you're allowed to use variables that are defined globally on this occasion. You should think about which objects that have been previously defined has a list of class labels.


```python
def find_class_idx(label):
    for i in range(len(network.classes)):
        if label == network.classes[i]:
            return i

    raise NotImplementedError()
```

#### 7) Slice Tennis Ball Class

Using the above function to find the correct index for tennis ball, you should implement a function to slice the calculated probability for tennis ball from the 1000 class probabilities calculated by the network. It should also convert the probability from MXNet `NDArray` to a NumPy `float32`.

We'll use this for our confidence score that the image is a tennis ball.


```python
def slice_tennis_ball_class(pred_probas):
    tennis_prob = pred_probas[find_class_idx('tennis ball')]
    return tennis_prob.astype('float32').asscalar()

    raise NotImplementedError()
```

#### 8) Classify Tennis Ball Images

We'll finish this assignment by bringing all of the components together and creating a `TennisBallClassifier` to classify images. You should implement the entire classification pipeline inside the `classify` function using the functions defined earlier on in the assignment. You should notice that the pre-trained model is loaded once during initialization, and then it should be used inside the `classify` method.


```python
class TennisBallClassifier():
    def __init__(self):
        self._network = load_pretrained_classification_network()
        
    def classify(self, filepath):
        # YOUR CODE HERE
        image = load_image(filepath)
        transformed_image = transform_image(image)
        self._visualize(transformed_image)
        probabilities = predict_probabilities(self._network, transformed_image)
        pred_proba = slice_tennis_ball_class(probabilities)   
        print('{0:.2%} confidence that image is a tennis ball.'.format(pred_proba))
        return pred_proba
    
    def _visualize(self, transformed_image):
        chw_image = transformed_image[0].transpose((1,2,0))
        chw_image = ((chw_image * 64) + 128).clip(0, 255).astype('uint8')
        plt.imshow(chw_image.asnumpy())
```

```python
classifier = TennisBallClassifier()
```


```python
filepath = Path(M3_IMAGES, 'erik-mclean-D23_XPbsx-8-unsplash.jpg')
pred_proba = classifier.classify(filepath)
np.testing.assert_almost_equal(pred_proba, 2.0355723e-05, decimal=3)
```

    0.00% confidence that image is a tennis ball.



![png](output_25_1.png)



```python
filepath = Path(M3_IMAGES, 'marvin-ronsdorf-CA998Anw2Lg-unsplash.jpg')
pred_proba = classifier.classify(filepath)
np.testing.assert_almost_equal(pred_proba, 0.9988895, decimal=3)
```

    99.92% confidence that image is a tennis ball.



![png](output_26_1.png)

This post was assembled by following the lecture from [AWS Computer Vision: Getting Started with GluonCV](https://www.coursera.org/learn/aws-computer-vision-gluoncv/home/welcome) course in [Coursera](https://www.coursera.org)
