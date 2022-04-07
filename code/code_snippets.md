# Python Code Snippets

<!-- MarkdownTOC -->

- Show samples from each class
- Display multiple images in one figure
- Plot images side by side
- Visualize a batch of image data
- Decorators
- Utility Classes
- References

<!-- /MarkdownTOC -->


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


## Display multiple images in one figure

```py
    # import libraries
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

## Plot images side by side

```py
    _, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()
```


## Visualize a batch of image data

TODO: Add code sample



## Decorators

```py
    def timer(func):
      """
      Display time it took for our function to run. 
      """
      @wraps(func)
      def wrapper(*args, **kwargs):
        start = time.perf_counter()
    
        # Call the actual function
        res = func(*args, **kwargs)
    
        duration = time.perf_counter() - start
        print(f'[{wrapper.__name__}] took {duration * 1000} ms')
        return res
        return wrapper
```

```py
    @timer
    def isprime(number: int):
      """ Check if a number is a prime number """
      isprime = False
      for i in range(2, number):
        if ((number % i) == 0):
          isprime = True
          break
          return isprime
```


```py
    def performance_check(func):
        """ Measure performance of a function """
        @wraps(func)
        def wrapper(*args, **kwargs):
          tracemalloc.start()
          start_time = time.perf_counter()
          res = func(*args, **kwargs)
          duration = time.perf_counter() - start_time
          current, peak = tracemalloc.get_traced_memory()
          tracemalloc.stop()
    
          print(f"\nFunction:             {func.__name__} ({func.__doc__})"
                f"\nMemory usage:         {current / 10**6:.6f} MB"
                f"\nPeak memory usage:    {peak / 10**6:.6f} MB"
                f"\nDuration:             {duration:.6f} sec"
                f"\n{'-'*40}"
          )
          return res
          return wrapper
```

```py
    @performance_check
    def is_prime_number(number: int):
        """Check if a number is a prime number"""
        # ....rest of the function
```


```py
    def repeater(iterations:int=1):
      """ Repeat the decorated function [iterations] times """
      def outer_wrapper(func):
        def wrapper(*args, **kwargs):
          res = None
          for i in range(iterations):
            res = func(*args, **kwargs)
          return res
        return wrapper
        return outer_wrapper
```

```py
    @repeater(iterations=2)
    def sayhello():
      print("hello")
```


```py
    def prompt_sure(prompt_text:str):
      """ Show prompt asking you whether you want to continue. Exits on anything but y(es) """
      def outer_wrapper(func):
        def wrapper(*args, **kwargs):
          if (input(prompt_text).lower() != 'y'):
            return
          return func(*args, **kwargs)
        return wrapper
        return outer_wrapper
```

```py
    @prompt_sure('Sure? Press y to continue, press n to stop')
    def say_hi():
      print("hi")
```


```py
    def trycatch(func):
      """ Wraps the decorated function in a try-catch. If function fails print out the exception. """
      @wraps(func)
      def wrapper(*args, **kwargs):
        try:
          res = func(*args, **kwargs)
          return res
        except Exception as e:
          print(f"Exception in {func.__name__}: {e}")
          return wrapper
```

```py
    @trycatch
    def trycatchExample(numA:float, numB:float):
      return numA / numB
```


## Utility Classes

```py
    from enum import Enum
    
    class Season(Enum):
        SPRING = 1
        SUMMER = 2
        FALL = 3
        WINTER = 4
```

```py
    spring = Season.SPRING
    spring.name
    spring.value
    
    fetched_season_value = 2
    matched_season = Season(fetched_season_value)
    matched_season
    # <Season.SUMMER: 2>
    
    list(Season)
    
    [x.name for x in Season]
```


```py
    from dataclasses import dataclass
    
    @dataclass
    class Student:
        name: str
        gender: str
```

```py
    student = Student("John", "M")
    student.name
    student.gender
    
    repr(student)   # __repr__
    # "Student(name='John', gender='M')"
    
    print(student)  # __str__
    Student(name='John', gender='M')
```

"""
# References

[4 simple tips for plotting multiple graphs in Python](https://towardsdatascience.com/4-simple-tips-for-plotting-multiple-graphs-in-python-38df2112965c)

[5 real handy python decorators for analyzing/debugging your code](https://towardsdatascience.com/5-real-handy-python-decorators-for-analyzing-debugging-your-code-c22067318d47)

[3 Alternatives for Regular Custom Classes in Python](https://betterprogramming.pub/3-alternatives-for-regular-custom-classes-in-python-2f2bafd66338)

[6 Must-Know Methods in Python’s Random Module](https://medium.com/geekculture/6-must-know-methods-in-pythons-random-module-338263b5f927)
"""