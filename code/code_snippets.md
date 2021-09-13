# Python Code Snippets

## Show samples from each class

```py
import numpy as np
import matplotlib.pyplot as plt

def show_images(num_classes):
    """
    Show image samples from each class
    """
    fig = plt.figure(figsize=(8,3))

    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        x_idx = X_train[idx,::]
        img_num = np.random.randint(x_idx.shape[0])
        im = np.transpose(x_idx[img_num,::], (1, 2, 0))
        ax.set_title(class_names[i])
        plt.imshow(im)

    plt.show()

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  X_train.shape
num_test, _, _, _ =  X_train.shape
num_classes = len(np.unique(y_train))

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

show_images(num_classes)
```


## How to Display Multiple Images in One Figure in Matplotlib

```py
#import libraries
import cv2
from matplotlib import pyplot as plt
  
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
num_rows = 2
num_cols = 2

# Read the images into list
images = []
img = cv2.imread('Image1.jpg')
images.append(img)

img = cv2.imread('Image2.jpg')
images.append(img)

img = cv2.imread('Image3.jpg')
images.append(img)

img = cv2.imread('Image4.jpg')
images.append(img)

  
# Adds a subplot at the 1st position
fig.add_subplot(num_rows, num_cols, 1)
  
# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("First")
  
# Adds a subplot at the 2nd position
fig.add_subplot(num_rows, num_cols, 2)
  
# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("Second")
  
# Adds a subplot at the 3rd position
fig.add_subplot(num_rows, num_cols, 3)
  
# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("Third")
  
# Adds a subplot at the 4th position
fig.add_subplot(num_rows, num_cols, 4)
  
# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("Fourth")
```


## Plot images side by side using matplotlib

```py
_, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
axs = axs.flatten()
for img, ax in zip(imgs, axs):
    ax.imshow(img)
plt.show()
```

## Visualize a batch of image data

TODO: Add code sample

