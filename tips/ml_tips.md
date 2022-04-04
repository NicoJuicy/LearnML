# Machine Learning Tips

<!-- MarkdownTOC -->

- Avoid Using Different Library Versions
- What aspect ratio to use for line plots?
  - Calculating the aspect ratio
  - Best practices
- Run ML model training/evaluation with TMUX
- Watch your training and GPU resources
- Testing the online inference models
  - A/B test
- Monitoring the model
- Improve Python Performance
- Improve Tensorflow Performance
  - Mixed Precision on NVIDIA GPUs
  - Mix Precision in Tensorflow
  - Fusing multiple ops into one
  - Fusion with Tensorflow 2.x
- Keras GPU Performance
- References

<!-- /MarkdownTOC -->


## Avoid Using Different Library Versions

A mistake we might run into is to use different versions of the various exploited libraries during the train / test and deployment phase.

The risk of using different versions is to have unexpected behaviours which may lead to wrong predictions.

A possible solution to this problem could be to create a virtual environment and install all the necessary libraries, also specifying the versions to be used and then use this virtual environment both during the train/test phase and during the deployment phase.



## What aspect ratio to use for line plots?

One of the most overlooked aspects of creating charts is the use of correct aspect ratios. 

### Calculating the aspect ratio

The concept of banking to 45 degrees is used to have coherency between the information presented and information perceived. 

Thus, we need to make sure that the orientation of the line segments in the chart is as close as possible to a slope of 45 degrees.

Here, the median absolute slope banking method has been used to calculate the aspect ratio for the sunspots plot. ￼

The `ggthemes` package provides a function called bank_slopes() to calculate the aspect ratio of the plot which takes x and y values as the two arguments. The default method is the median absolute slope banking. 

### Best practices

- **Plotting multiple line graphs for comparison on a single chart:** The default aspect ratio works only if you do not plan to compare two different plots.

- **Comparing different line graphs from different charts:** Make sure the aspect ratio for each plot remains the same. Otherwise, the visual interpretation will be skewed. 

  1. Using incorrect or default aspect ratios: In this case, we choose the aspect ratios such that the plots end up being square-shaped.

  2. Calculating aspect ratios per plot: The best approach to compare the plots is to calculate the aspect ratios for each plot. 

**Time-series:** It is best to calculate the aspect ratio since some hidden information can be more pronounced when using the correct aspect ratio for the plot.


## Run ML model training/evaluation with TMUX

`tmux` can be used when you want to detach processes from their controlling terminals which allows remote sessions to remain active without beingvisible.


## Watch your training and GPU resources

```bash
  watch -n nvidia-smi
  nvtop
  gpustat
```


## Testing the online inference models

ML system testing is also more complex a challenge than testing manually coded systems, due to the fact that ML system behavior depends strongly on data and models that cannot be strongly specified a priori.

Figure: The Machine Learning Test Pyramid.

ML requires more testing than traditional software engineering.

### A/B test

To measure the impact of a new model, we need to augment the evaluation by running statistical A/B tests. 

In an A/B test, users are split into two distinct non-overlapping cohorts. To run an A/B test, the population of users must be split into statistically identical populations that each experience a different algorithm.


## Monitoring the model

Once a model has been deployed its behavior must be monitored. 

A model’s predictive performance is expected to degrade over time as the environment changes callef concept drift which occurs when the distributions of the input features or output target shift away from the distribution upon which the model was originally trained.

When concept drift has been detected, we need to retrain the ML model but detecting drift can difficult.

One strategy for monitoring is to use a metric from a deployed model that can be measured over time such as measuring the output distribution. The observed distribution can be compared to the training output distribution, and alerts can notify data scientists when the two quantities diverge.

Popular ML/AI deployment tools: TensorFlow Serving, MLflow, Kubeflow, Cortex, Seldon.io, BentoML, AWS SageMaker, Torchserve, Google AI.


## Improve Python Performance

The rule of thumb is that a computer is a sum of its parts or weakest link. In addition, the basic performance equation reminds us that there are always tradeoffs in hardware/software performance. 

Thus, there is no silver bullet hardware or software technology that will magically improve computer performance.

Hardware upgrades (vertical scaling) usually provide only marginal improvement in performance. However, we can achieve as much as 30-100x performance improvement using software libraries and code refactoring to improve parallelization (horizontal scaling) [1][2].

> There is no silver bullet to improve performance.

In general, improving computer performance is a cumulative process of several or many different approaches primarily software related.

> NOTE: Many of the software libraries to improve pandas performance also enhance numpy performance as well. 


[Intel Extension for Scikit-learn](https://intel.github.io/scikit-learn-intelex/index.html#intelex)

[Installing Intel Distribution for Python and Intel Performance Libraries with Anaconda](https://www.intel.com/content/www/us/en/developer/articles/technical/using-intel-distribution-for-python-with-anaconda.html)

[Intel Optimization for TensorFlow Installation Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html)

[How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)

```py
  # Install intel conda packages with Continuum's Python (version conflicts on Linux)
  conda install mkl intel::mkl --no-update-deps
  conda install numpy intel::numpy --no-update-deps

  # macOS: AttributeError: module 'numpy' has no attribute 'ndarray'

  # Needed on macOS
  conda install -c intel numpy=1.19.5 --no-update-deps

  # Install intel optimization for tensorflow from anaconda channel 
  # cannot install tensorflow-mkl on macOS (version conflicts)
  conda install -c anaconda tensorflow
  conda install -c anaconda tensorflow-mkl

  # Install intel optimization for tensorflow from intel channel
  # conda install tensorflow -c intel
```

```py
  # Intel Extension for Scikit-learn
  conda install -c conda-forge scikit-learn-intelex

  from sklearnex import patch_sklearn
  patch_sklearn()
```


## Improve Tensorflow Performance

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models without almost no source code changes.

XLA compiles the TensorFlow graph into a sequence of computation kernels generated specifically for the given model. 

Without XLA, TensorFlow graph executes three kernels: one for addition, one for multiplication, and one for reduction. However, XLA compiles these three kernels into one kernel so that intermediate results no longer have to be saved during the computation.

Using XLA, we can use less memory also speed up training.

Enabling XLA is as simple as using the  `@tf.function` decorator. 

```py
  # enable XlA globally
  tf.config.optimizer.set_jit(True)
  
  @tf.function(jit_compile=True)
  def dense_layer(x, w, b):
      return add(tf.matmul(x, w), b)
```

[The Ultimate TensorFlow-GPU Installation Guide For 2022 And Beyond](https://towardsdatascience.com/the-ultimate-tensorflow-gpu-installation-guide-for-2022-and-beyond-27a88f5e6c6e)

[Leverage the Intel TensorFlow Optimizations for Windows to Boost AI Inference Performance](https://medium.com/intel-tech/leverage-the-intel-tensorflow-optimizations-for-windows-to-boost-ai-inference-performance-ba56ba60bcc4)

[Time to Choose TensorFlow Data over ImageDataGenerator](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435)

[Optimizing a TensorFlow Input Pipeline: Best Practices in 2022](https://medium.com/@virtualmartire/optimizing-a-tensorflow-input-pipeline-best-practices-in-2022-4ade92ef8736)


### Mixed Precision on NVIDIA GPUs

Mixed precision training offers significant computational speedup by performing operations in the half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.

There are numerous benefits to using numerical formats with lower precision than 32-bit floating-point: require less memory; require less memory bandwidth. 

- Speeds up math-intensive operations, such as linear and convolution layers by using Tensor Cores.

- Speeds up memory-limited operations by accessing half the bytes compared to single-precision.

- Reduces memory requirements for training models, enabling larger models or larger mini-batches.

Among NVIDIA GPUs, those with compute capability 7.0 or higher will see the greatest performance benefit from mixed-precision because they have special hardware units called Tensor Cores to accelerate float16 matrix multiplications and convolutions.


### Mix Precision in Tensorflow

The mixed precision API is available in TensorFlow 2.1 with Keras interface. 

To use mixed precision in Keras, we have to create a _dtype policy_ which specify the dtypes layers will run in. 

Then, layers created will use mixed precision with a mix of float16 and float32.

```py
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)
  # Now design your model and train it
```

> NOTE: Tensor Cores which provide mix precision, requires certain dimensions of tensors such as dimensions of your dense layer, number of filters in Conv layers, number of units in RNN layer to be a multiple of 8.

To compare the performance of mixed-precision with float32, change the policy from `mixed_float16` to float32 which can improve performance up to 3x.

### Fusing multiple ops into one

Usually when you run a TensorFlow graph, all operations are executed individually by the TensorFlow graph executor which means each op has a pre-compiled GPU kernel implementation. 

Fused Ops combine operations into a single kernel for improved performance.

Without fusion, without XLA, the graph launches three kernels: one for the multiplication, one for the addition and one for the reduction.

```py
  def model_fn(x, y, z): 
      return tf.reduce_sum(x + y * z)
```

Using op fusion, we can compute the result in a single kernel launch by fusing the addition, multiplication, and reduction into a single GPU kernel.

### Fusion with Tensorflow 2.x

Newer Tensorflow versions come with XLA which does fusion along with other optimizations for us.

Fusing ops together provides several performance advantages:

- Completely eliminates Op scheduling overhead (big win for cheap ops)

- Increases opportunities for ILP, vectorization etc.

- Improves temporal and spatial locality of data access




## Keras GPU Performance

[Using GPUs With Keras: A Tutorial With Code](https://wandb.ai/authors/ayusht/reports/Using-GPUs-With-Keras-A-Tutorial-With-Code--VmlldzoxNjEyNjE)

[Use a GPU with Tensorflow](https://www.tensorflow.org/guide/gpu)

[Using an AMD GPU in Keras](https://www.petelawson.com/post/using-an-amd-gpu-in-keras/)

[TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)](https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su)


```py
    import os
    
    from tensorflow.python.client import device_lib
    
    # disable oneDNN warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    print(f"tensorflow version is {tf.__version__}")

    # Check if tensorflow is using GPU
    print(f"\nPhysical Devices:\n{tf.config.list_physical_devices('GPU')}")
    print(f"\n\nLocal Devices:\n{device_lib.list_local_devices()}")
    print(f"\nNum GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
```

```py
    import tensorflow as tf
    import tensorflow_datasets as tfds

    from tensorflow.python.framework.ops import disable_eager_execution
    from tensorflow.python.compiler.mlcompute import mlcompute

    #disable_eager_execution()

    mlcompute.set_mlc_device(device_name='gpu')
```


[Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)

[GPU-Accelerated Machine Learning on MacOS](https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545)

[apple/tensorflow_macos](https://github.com/apple/tensorflow_macos/issues/153)

[tensorflow Mac OS gpu support](https://stackoverflow.com/questions/44744737/tensorflow-mac-os-gpu-support)

[Install Tensorflow 2 and PyTorch for AMD GPUs](https://medium.com/analytics-vidhya/install-tensorflow-2-for-amd-gpus-87e8d7aeb812)

[Installing TensorFlow on the M1 Mac](https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776)




## References

[1] [Four Reasons to Love Ray and GPU-accelerated Distributed Compute](https://www.kdnuggets.com/2022/02/domino-four-reasons-love-ray-gpu-accelerated-distributed-compute.html)

[2] [How we optimized Python API server code 100x](https://towardsdatascience.com/how-we-optimized-python-api-server-code-100x-9da94aa883c5)

[3] [Accelerate your training and inference running on Tensorflow](https://towardsdatascience.com/accelerate-your-training-and-inference-running-on-tensorflow-896aa963aa70)

[4] [A simple guide to speed up your training in TensorFlow](https://blog.seeso.io/a-simple-guide-to-speed-up-your-training-in-tensorflow-2-8386e6411be4?gi=55c564475d16

[Best practices in the deployment of AI models](https://nagahemachandchinta.medium.com/best-practices-in-the-deployment-of-ai-models-c929c3146416)

[Data Science Mistakes to Avoid: Data Leakage](https://towardsdatascience.com/data-science-mistakes-to-avoid-data-leakage-e447f88aae1c)

[10 Simple Things to Try Before Neural Networks](https://www.kdnuggets.com/2021/12/10-simple-things-try-neural-networks.html)

[What aspect ratio to use for line plots](https://towardsdatascience.com/should-you-care-about-the-aspect-ratio-when-creating-line-plots-ed423a5dceb3)

[Introduction to TensorFlow Probability (Bayesian Neural Network)](https://towardsdatascience.com/introduction-to-tensorflow-probability-6d5871586c0e)



