# GAN Tips

Since I completed my Master's Thesis on Generative Adversarial Networks (GANs), I thought I would share a few things that I have learned:

1. What is your goal/application? GAN is primarily used for computer vision applications. If you try to use GAN with custom datatsets (such as financial data), you will most likely encounter formatting and other internal model issues with PyTorch. In fact, there are newer, better models.

2. GAN is best used for applications in which the models need to automate the process to "fine-tune" the model. For example, I was able to achieve 98%+ accuracy using a LSTM model by choosing the right combination of technical indicators given in the research literature, so GAN could hardly improve on these results.

If you are a student and have a .edu email account, you can get free access to [OReilly] (https://www.oreilly.com/) which has some good books on the topic that I found useful:

- PyTorch Artificial Intelligence Fundamentals
- PyTorch Computer Vision Cookbook
- Mastering PyTorch
- Generative Adversarial Networks Projects


Here are some online resources that I found helpful:

- [How to Identify and Diagnose GAN Failure Modes](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)

- [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

- [Deep Convolutional vs Wasserstein Generative Adversarial Network](https://towardsdatascience.com/deep-convolutional-vs-wasserstein-generative-adversarial-network-183fbcfdce1f)


Here are some PyTorch resources that I found helpful:

- [GAN in PyTorch](https://jaketae.github.io/study/pytorch-gan/)

- [PyTorch Tutorials](https://github.com/yunjey/pytorch-tutorial/tree/0500d3df5a2a8080ccfccbc00aca0eacc21818db)

- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)


- [How to Build a DCGAN with PyTorch](https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a)


## Generating Synthetic Data

GANs can generate several types of synthetic data, including image data, tabular data, and sound/speech data.

[Gretel Synthetics](https://github.com/gretelai/gretel-synthetics)

[Creating synthetic time series data](https://gretel.ai/blog/creating-synthetic-time-series-data)

[Synthetic data generation using Generative Adversarial Networks (GANs)](https://medium.com/data-science-at-microsoft/synthetic-data-generation-using-generative-adversarial-networks-gans-part-1-47ecbf46b575)

[Generating Synthetic Data Using a Generative Adversarial Network (GAN) with PyTorch](https://visualstudiomagazine.com/articles/2021/06/02/gan-pytorch.aspx?m=1)



Good luck!