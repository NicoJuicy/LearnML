## Types of Parallelization

Parallelization is possible in two ways:

- Multithreading: Using multiple threads of a process/worker

- Multiprocessing: Using multiple processors

Multithreading is useful for I/O bound applications. For example, when we have to download and upload multiple files.

Multiprocessing is useful for CPU-bound applications. You can consider the  example below as one of the use cases of multiprocessing.

Suppose we have 1000 images saved in a folder, and for each image, I have to perform the following operations.

- Convert image to grayscale
- Resize the grayscale image to a given size
- Save the modified image in a folder

Doing this process on each image is independent of each other -- processing one image would not affect any other image in the folder. 

Therefore, multiprocessing can help us reduce the total time. 

The total time will be reduced by a factor equal to the number of processors we use in parallel. This is one of many examples where you can use parallelization to save time.


## References

[Parallelize your python code to save time on data processing](https://towardsdatascience.com/parallelize-your-python-code-to-save-time-on-data-processing-805934b826e2)

