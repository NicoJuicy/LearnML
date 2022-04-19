# Python Code Snippets

<!-- MarkdownTOC levels=1,2,3 -->

- Show samples from each class
- Display multiple images in one figure
- Plot images side by side
- Visualize a batch of image data
- The Decorator Pattern
    - @staticmethod
    - @classmethod
    - @property
- Decorator Code Snippets
    - Timer
    - Measure Function Performance
    - Repeat
    - Show prompt
    - Try/Catch
- Python one-liners
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

----------

## The Decorator Pattern

### @staticmethod

A static method is a method that does not require the creation of an instance of a class. 

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        def get_number(self):
            return self.number
          
        @staticmethod
        def get_emergency_number():
            return "911"
          
    Cellphone.get_emergency_number()
    # '911'
```

### @classmethod

A class method requires the class itself as the first argument which is written as cls. 

A class method normally works as a factory method and returns an instance of the class with supplied arguments. However, it does not have to work as a factory class and return an instance.

We can create an instance in the class method and do whatever you need without having to return it.

Class methods are very commonly used in third-party libraries.

Here, it is a factory method here and returns an instance of the Cellphone class with the brand preset to “Apple”.

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        def get_number(self):
            return self.number
          
        @staticmethod
        def get_emergency_number():
            return "911"
          
        @classmethod
        def iphone(cls, number):
            _iphone = cls("Apple", number)
            print("An iPhone is created.")
            return _iphone
         
    iphone = Cellphone.iphone("1112223333")
    # An iPhone is created.
    iphone.get_number()
    # "1112223333"
    iphone.get_emergency_number()
    # "911"
```

If you use class methods properly, you can reduce code redundancy dramatically and make your code more readable and more professional. 

The key idea is that we can create an instance of the class based on some specific arguments in a class method, so we do not have to repeatedly create instances in other places (DRY).


### @property

In the code snippet above, there is a function called `get_number` which returns the number of a Cellphone instance. 

We can optimize the method a bit and return a formatted phone number.


In Python, we can also use getter and setter to easily manage the attributes of the class instances.


```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        @property
        def number(self):
            _number = "-".join([self._number[:3], self._number[3:6],self._number[6:]])
            return _number
        
        @number.setter
        def number(self, number):
            if len(number) != 10:
                raise ValueError("Invalid phone number.")
            self._number = number

    cellphone = Cellphone("Samsung", "1112223333")
    print(cellphone.number)
    # 111-222-3333

    cellphone.number = "123"
    # ValueError: Invalid phone number.
```


Here is the complete example using the three decorators in Python: `@staticmethod`, `@classmethod`, and `@property`:

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        @property
        def number(self):
            _number = "-".join([self._number[:3], self._number[3:6],self._number[6:]])
            return _number

        @number.setter
        def number(self, number):
            if len(number) != 10:
                raise ValueError("Invalid phone number.")
            self._number = number
        
        @staticmethod
        def get_emergency_number():
            return "911"
        
        @classmethod
        def iphone(cls, number):
            _iphone = cls("Apple", number)
            print("An iPhone is created.")
            return _iphone
```


## Decorator Code Snippets

### Timer

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

### Measure Function Performance

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

### Repeat

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

### Show prompt

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

### Try/Catch

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


----------


## Python one-liners

```py
    # Palindrome Python One-Liner
    phrase.find(phrase[::-1])

    # Swap Two Variables Python One-Liner
     a, b = b, a

    # Sum Over Every Other Value Python One-Liner
     sum(stock_prices[::2])

    # Read File Python One-Liner
     [line.strip() for line in open(filename)]

    # Factorial Python One-Liner
     reduce(lambda x, y: x * y, range(1, n+1))

    # Performance Profiling Python One-Liner
     python -m cProfile foo.py

    # Superset Python One-Liner
     lambda l: reduce(lambda z, x: z + [y + [x] for y in z], l, [[]])

    # Fibonacci Python One-Liner
     lambda x: x if x<=1 else fib(x-1) + fib(x-2)

    # Quicksort Python One-liner
     lambda L: [] if L==[] else qsort([x for x in L[1:] if x< L[0]]) + L[0:1] + qsort([x for x in L[1:] if x>=L[0]])

    # Sieve of Eratosthenes Python One-liner
    reduce( (lambda r,x: r-set(range(x**2,n,x)) if (x in r) else r), range(2,int(n**0.5)), set(range(2,n)))
```

---------


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



## References

[4 simple tips for plotting multiple graphs in Python](https://towardsdatascience.com/4-simple-tips-for-plotting-multiple-graphs-in-python-38df2112965c)

[5 real handy python decorators for analyzing/debugging your code](https://towardsdatascience.com/5-real-handy-python-decorators-for-analyzing-debugging-your-code-c22067318d47)

[3 Alternatives for Regular Custom Classes in Python](https://betterprogramming.pub/3-alternatives-for-regular-custom-classes-in-python-2f2bafd66338)

[How to Use the Magical @staticmethod, @classmethod, and @property Decorators in Python](https://betterprogramming.pub/how-to-use-the-magical-staticmethod-classmethod-and-property-decorators-in-python-e42dd74e51e7?gi=8734ec8451fb)

[Learn Python By Example: 10 Python One-Liners That Will Help You Save Time](https://medium.com/@alains/learn-python-by-example-10-python-one-liners-that-will-help-you-save-time-ccc4cabb9c68)


[6 Must-Know Methods in Python’s Random Module](https://medium.com/geekculture/6-must-know-methods-in-pythons-random-module-338263b5f927)
