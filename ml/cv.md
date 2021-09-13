# Computer Vision Notes

[A Gentle Introduction to Computer Vision](https://machinelearningmastery.com/what-is-computer-vision/)

[Guide to Deep Learning for Computer Vision](https://machinelearningmastery.com/start-here/#dlfcv)

Computer Vision (CV) is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos.

Many popular computer vision applications involve trying to recognize things in photographs:

- Object Classification: What broad category of object is in this photograph?

— Object Identification: Which type of a given object is in this photograph?

- Object Verification: Is the object in the photograph?

- Object Detection: Where are the objects in the photograph?

- Object Landmark Detection: What are the key points for the object in the photograph?

- Object Segmentation: What pixels belong to the object in the image?

- Object Recognition: What objects are in this photograph and where are they?


## Image Data Handling

[How to Load and Manipulate Images for Deep Learning in Python With PIL/Pillow](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/)

[How to Load and Visualize Standard Computer Vision Datasets With Keras](https://machinelearningmastery.com/how-to-load-and-visualize-standard-computer-vision-datasets-with-keras/)

[How to Load, Convert, and Save Images With the Keras API](https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/)

### NHWC vs HCHW

[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)

In the past, I encountered obscure image format issues going between Linux and macOS (NHWC vs HCHW).


## Image Data Preparation

[Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer/)

When training vision models, it is common to resize images to a lower dimension ((224 x 224), (299 x 299), etc.) to allow mini-batch learning and also to keep up the compute limitations. 

We generally make use of image resizing methods like bilinear interpolation for this step and the resized images do not lose much of their perceptual character to the human eyes.

In “Learning to Resize Images for Computer Vision Tasks,” Talebi et al. show that if we try to optimize the perceptual quality of the images for the vision models rather than the human eyes, their performance can further be improved.

**For a given image resolution and a model, how to best resize the given images?**

As shown in the paper, this idea helps to consistently improve the performance of the common vision models (pre-trained on ImageNet-1k) such as DenseNet-121, ResNet-50, MobileNetV2, and EfficientNets. 

In the example, we will implement the learnable image resizing module as proposed in the paper and demonstrate that on the Cats and Dogs dataset using the DenseNet-121 architecture.


[How to Manually Scale Image Pixel Data for Deep Learning](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)

[How to Normalize, Center, and Standardize Images in Keras](https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/)

[How to Evaluate Pixel Scaling Methods for Image Classification](https://machinelearningmastery.com/how-to-evaluate-pixel-scaling-methods-for-image-classification/)



## Image Data Augmentation

[How to Load Large Datasets From Directories](https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/)

[How to Configure and Use Image Data Augmentation](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

[Introduction to Test-Time Data Augmentation](https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/)

## Object Recognition

[A Gentle Introduction to Object Recognition](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

[How to Perform Object Detection with Mask R-CNN](https://machinelearningmastery.com/how-to-perform-object-detection-in-photographs-with-mask-r-cnn-in-keras/)

[How to Perform Object Detection With YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)

## Image Classification

[How to Develop a CNN for CIFAR-10 Photo Classification](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)

[How to Develop a CNN to Classify Photos of Dogs and Cats](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)

[How to Develop a CNN to Classify Satellite Photos](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)

## Basics of Convolutional Neural Networks

[Gentle Introduction to Convolutional Layers in CNNS](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

[Gentle Introduction to Padding and Stride in CNNs](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)

[Gentle Introduction to Pooling Layers in CNNs](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)





## Reading an image

```py
def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
```

## Showing multiple images in a grid

```py
image_paths = list(Path('./dogs').iterdir()) 
images = [read_image(p) for p in image_paths]

fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 4

pics = []
for i in range(columns*rows):
    pics.append(fig.add_subplot(rows, columns, i+1,title=image_paths[i].parts[-1].split('.')[0]))
    plt.imshow(images[i])

plt.show()
```

## Object detection and tracking

The approach will depend on your goal/project. I have mostly seen classification (object detection) apps which try to determine the counts of types of objects (pedestrian/vehicle). 

**NOTE:** Be sure to open the articles in “private” tab/window. 

[Making Road Traffic Counting App based on Computer Vision and OpenCV](https://medium.com/machine-learning-world/tutorial-making-road-traffic-counting-app-based-on-computer-vision-and-opencv-166937911660)

[Vehicle Detection and Tracking using Machine Learning and HOG](https://towardsdatascience.com/vehicle-detection-and-tracking-using-machine-learning-and-hog-f4a8995fc30a?gi=b793ee27f135)

Yolo is the latest approach to object detection and there is a version that can run on raspi. Yolo is supposed to be easier to setup/use and faster performance.  

[Object tracking using YOLOv4 and TensorFlow](https://pythonawesome.com/object-tracking-implemented-with-yolov4-and-tensorflow/)


—————————-


## [Machine Learning: Improving Classification accuracy on MNIST using Data Augmentation](https://towardsdatascience.com/improving-accuracy-on-mnist-using-data-augmentation-b5c38eb5a903?gi=916228e35c66)

We can write a method to shift the images in all four directions by the given order.

We will shift the images to each of the four directions by one pixel and generate four more images from a single image.



## [How To Augment the MNIST Dataset Using Tensorflow](https://medium.com/the-data-science-publication/how-to-augment-the-mnist-dataset-using-tensorflow-4fbf113e99a0)

### Step 2. Identify and Plot Baseline Digits Using Matplotlib

We plot a subset of the MNIST images to help us understand the augmentation effects on the MNIST dataset. 

To plot a subset of MNIST images, use the following code:

### Step 3. Understand Image Augmentation and Techniques Relevant To MNIST

The original MNIST dataset contains centered, upright, and size normalized digits. 

Realistically, hand-written digits will seldom meet these criteria in real-world applications. Some digits will be larger, smaller, rotated, or skewed more than others. 

To create a robust digit recognition model, it is in your interest to augment the MNIST dataset and capture these types of behavior. 

We discuss the various types of augmentation techniques we can use to enhance the MNIST digit dataset. 

In this tutorial, we will use the ImageDataGenerator class available in the Tensorflow.Keras python library. 

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


### Augment The MNIST Dataset

Finally, we can combine all of the previously mentioned transformations to obtain unique digit representations that can now be used to improve digit recognition model performance.


[Image Processing and Data Augmentation Techniques for Computer Vision](https://towardsdatascience.com/image-processing-techniques-for-computer-vision-11f92f511e21)

[Data Augmentation Compilation with Python and OpenCV](https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0)

[5 Image Augmentation Techniques Using imgAug](https://betterprogramming.pub/5-common-image-augmentations-for-machine-learning-c6b5a03ebf38)

[5 Useful Image Manipulation Techniques Using Python OpenCV](https://betterprogramming.pub/5-useful-image-manipulation-techniques-using-python-opencv-505492d077ef)


## VGG 

Given a photograph of an object, derrmine which of 1,000 specific objects the photograph shows.

A competition-winning model for this task is the VGG model by researchers at Oxford. 

What is important about this model (besides capability of classifying objects in photographs) is that the model weights are freely available and can be loaded and used in your own models and applications.

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is an annual computer vision competition. Each year, teams compete on two tasks. 

The first is to detect objects within an image coming from 200 classes which is called _object localization_. 

The second is to classify images, each labeled with one of 1000 categories which is called _image classification_. 

[VGG-16 CNN model](https://www.geeksforgeeks.org/vgg-16-cnn-model/)

[How to Develop VGG, Inception and ResNet Modules from Scratch in Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

[How to Develop VGG, Inception, and ResNet Modules from Scratch in Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)


## Residual Networks (ResNet)

[Residual Networks (ResNet)](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)

After the first CNN-based architecture (AlexNet) that win the ImageNet 2012 competition, every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate which works for less number of layers.  

When we increase the number of layers, there is a common problem in deep learning called Vanishing/Exploding gradient which causes the gradient to become 0 or too large. Thus, when we increase the number of layers, the training and test error rate also increases.

### Residual Block

In order to solve the problem of the vanishing/exploding gradient, ResNet introduced the concept called Residual Network which uses a technique called _skip connections_. 

The skip connection skips training from a few layers and connects directly to the output.



## References

[Achieving 95.42% Accuracy on Fashion-Mnist Dataset](https://secantzhang.github.io/blog/deep-learning-fashion-mnist)



