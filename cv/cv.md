# Computer Vision

[A Gentle Introduction to Computer Vision](https://machinelearningmastery.com/what-is-computer-vision/)

[Guide to Deep Learning for Computer Vision](https://machinelearningmastery.com/start-here/#dlfcv)

Computer Vision (CV) is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos.

Many popular computer vision applications involve trying to recognize things in photographs:

- Object Classification: What broad category of object is in this photograph?

- Object Identification: Which type of a given object is in this photograph?

- Object Verification: Is the object in the photograph?

- Object Detection: Where are the objects in the photograph?

- Object Landmark Detection: What are the key points for the object in the photograph?

- Object Segmentation: What pixels belong to the object in the image?

- Object Recognition: What objects are in this photograph and where are they?

## What is Image Recognition?

Image recognition or image classification is the task of recognizing images and classifying them in one of the several predefined individual classes.

Image recognition can perform different tasks including:

- Classification: It is the recognition of the “class” of an image. An image can only have a single class.

- Tagging: It also falls under the classification task but with a higher level of precision. It can identify the presence of numerous concepts or entities within an image. One or more tags can thus be allotted to a specific image.

- Detection: This is important when you want to find an entity in an image . Once the object is found, a bounding box is placed around the object in question.

- Segmentation: This falls under the detection task, and it is responsible for locating an element in an image to the nearest pixel. In some instances, it is necessary to maintain a higher degree of accuracy, just like in the development of autonomous cars.


## Overview

The following sections discuss several important CV concepts:

- Image Data Loading
- Image Data Preparation
- Image Data Augmentation

- Object Recognition
- Object Classification
- Object Detection and Tracking


## Image Data Loading

[How to Load and Manipulate Images for Deep Learning in Python With PIL/Pillow](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/)

[How to Load and Visualize Standard Computer Vision Datasets With Keras](https://machinelearningmastery.com/how-to-load-and-visualize-standard-computer-vision-datasets-with-keras/)

[How to Load, Convert, and Save Images With the Keras API](https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/)

### NHWC vs HCHW

[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)

In the past, I encountered obscure image format issues going between Linux and macOS (NHWC vs HCHW).


## Object Recognition

[A Gentle Introduction to Object Recognition](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

[How to Perform Object Detection with Mask R-CNN](https://machinelearningmastery.com/how-to-perform-object-detection-in-photographs-with-mask-r-cnn-in-keras/)

[How to Perform Object Detection With YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)


## Object Classification

[How to Develop a CNN for CIFAR-10 Photo Classification](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)

[How to Develop a CNN to Classify Photos of Dogs and Cats](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)

[How to Develop a CNN to Classify Satellite Photos](https:/low /machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)


## Basics of Convolutional Neural Networks

[Gentle Introduction to Convolutional Layers in CNNS](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

[Gentle Introduction to Padding and Stride in CNNs](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)

[Gentle Introduction to Pooling Layers in CNNs](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)



## Object Detection and Tracking

The approach will depend on your goal/project but most common object detection projects are classification (object detection) apps which try to determine the counts of types of objects (pedestrian/vehicle). 

**NOTE:** Be sure to open the articles in “private” tab/window. 

[Making Road Traffic Counting App based on Computer Vision and OpenCV](https://medium.com/machine-learning-world/tutorial-making-road-traffic-counting-app-based-on-computer-vision-and-opencv-166937911660)

[Vehicle Detection and Tracking using Machine Learning and HOG](https://towardsdatascience.com/vehicle-detection-and-tracking-using-machine-learning-and-hog-f4a8995fc30a?gi=b793ee27f135)

Yolo is the latest approach to object detection and there is a version that can run on raspi. Yolo is supposed to be easier to setup/use and faster performance.  

[Object tracking using YOLOv4 and TensorFlow](https://pythonawesome.com/object-tracking-implemented-with-yolov4-and-tensorflow/)



## Code Snippets

### Reading an image

```py
def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
```

### Showing multiple images in a grid

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


---------


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
