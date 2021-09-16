# Data Augmentation

[Improving Classification accuracy on MNIST using Data Augmentation](https://towardsdatascience.com/improving-accuracy-on-mnist-using-data-augmentation-b5c38eb5a903?gi=916228e35c66)

We can write a method to shift the images in all four directions by the given order.

We will shift the images to each of the four directions by one pixel and generate four more images from a single image.


## Image Data Pipeline

[Time to Choose TensorFlow Data over ImageDataGenerator](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435)

We can build better and faster image pipelines using `tf.data`. 

While training a neural network, it is quite common to use `ImageDataGenerator` class to generate batches of tensor image data with real-time data augmentation, but the `tf.data` API can be used to build a faster input data pipeline with reusable pieces.


----------



# Augment MNIST Dataset Using Tensorflow

[How To Augment the MNIST Dataset Using Tensorflow](https://medium.com/the-data-science-publication/how-to-augment-the-mnist-dataset-using-tensorflow-4fbf113e99a0)

### Step 1: Importing the MNIST dataset

In step 1, we will import the MNIST dataset using the tensorflow library. The imported dataset will be divided into train/test and input/output arrays.

```py
    from tensorflow.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

### Step 2: Identify and Plot Baseline Digits Using Matplotlib

We plot a subset of the MNIST images to help us understand the augmentation effects on the MNIST dataset. 

To plot a subset of MNIST images, use the following code:

### Step 3:  Understand Image Augmentation and Techniques Relevant To MNIST

The original MNIST dataset contains centered, upright, and size normalized digits. 

Realistically, hand-written digits will seldom meet these criteria in real-world applications. Some digits will be larger, smaller, rotated, or skewed more than others. 

To create a robust digit recognition model, it is in your interest to augment the MNIST dataset and capture these types of behavior. 

We discuss the various types of augmentation techniques we can use to enhance the MNIST digit dataset. 

In this tutorial, we will use the `ImageDataGenerator` class available in the `tensorflow.keras` python library. 

- Rotate
- Shift
- Shear
- Zoom

- Crop (center and random)
- Resize
- Flip (horiz/vert)
- ColorJitter
- Blur
- Greyscale

- Adding Noise
- Saturation
- Cutout
- Filter

_Cutout_ is a simple regularization technique of randomly masking out square regions of input during training which can be used to improve the robustness and overall performance of convolutional neural networks. 

This method can also be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.


_ColorJitter_ is another simple type of image data augmentation where we randomly change the brightness, contrast, and saturation of the image. 

### Overlay Images

Sometimes, we need to add a background to an existing image for formatting purposes. For instance, by padding a solid color as margins, we can make many images of different sizes become the same shape. Several techniques are relevant here.

### Step 4: Augment The MNIST Dataset

Finally, we can combine all of the previously mentioned transformations to obtain unique digit representations that can now be used to improve digit recognition model performance.



## References

[Image Processing and Data Augmentation Techniques for Computer Vision](https://towardsdatascience.com/image-processing-techniques-for-computer-vision-11f92f511e21)

[Data Augmentation Compilation with Python and OpenCV](https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0)

[5 Image Augmentation Techniques Using imgAug](https://betterprogramming.pub/5-common-image-augmentations-for-machine-learning-c6b5a03ebf38)

[5 Useful Image Manipulation Techniques Using Python OpenCV](https://betterprogramming.pub/5-useful-image-manipulation-techniques-using-python-opencv-505492d077ef)

[Achieving 95.42% Accuracy on Fashion-Mnist Dataset](https://secantzhang.github.io/blog/deep-learning-fashion-mnist)

