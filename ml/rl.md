# Reinforcement Learning

## Overview

[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning?wprov=sfti1) (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. 

The three basic machine learning paradigms are: supervised learning, unsupervised learning, and reinforcement learning. 

RL differs from supervised learning:

- RL does not need labeled input/output pairs. 

- RL focuses on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).

The environment is typically stated in the form of a _Markov decision process (MDP)_ since many RL algorithms use dynamic programming techniques. 


## What makes Reinforcement Learning so hard?

The main difference between the classical dynamic programming methods and RL algorithms is that RL does not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.

For every state, we want the ability to make a good decision. Decisions are made sequentially: we order a combination of products, observe the products sold, and at the end of the day we must be able to make a re-order decision for every possible scenario that may unfold.

Thus, _deterministic_ optimization provides decisions whereas _stochastic_ optimization provides decision-making policies. 

Deterministic optimization solves a single problem whereas stochastic optimization solves all problems that may arise. 

> RL does not tell us _what_ decisions to make but _how_ to make them. 


## Summary

- The goal of RL is to find a policy that works under any circumstance rather than just a decision which makes it much harder than deterministic optimization.

- RL is usually applied in noisy stochastic environments and ML algorithms have a tendency to fit the noise rather than the controllable patterns.

- Challenges in RL are the many different subfields (no uniform language or toolbox), problem representation in mathematical form, and the reliance on limited real-world observations to learn policies.


## References

Here are some articles covering some of the key RL concepts: `A*` search algorithm, Markov Decision Process (MDP), Partially Observable Environment, and Multi-Agent RL (MARL).

Some of the articles are a series covering several RL topics. 

There are also some good books on RL on [oreilly.com](https://www.oreilly.com/) that are free to access if you have .edu email account.


[Top Down View at Reinforcement Learning](https://towardsdatascience.com/top-down-view-at-reinforcement-learning-f4a8b35ebf9a?gi=c3f6a92209fd)

[The very basics of Reinforcement Learning](https://becominghuman.ai/the-very-basics-of-reinforcement-learning-154f28a79071?gi=49c5fe317a90)

[From A* to MARL](https://omrikaduri.medium.com/from-a-to-marl-part-1-mapf-d4c0796ce1af)

[An Introduction to Reinforcement Learning with OpenAI Gym, RLlib, and Google Colab](https://www.kdnuggets.com/2021/09/intro-reinforcement-learning-openai-gym-rllib-colab.html)

[Why Hasn’t Reinforcement Learning Conquered The World?](https://towardsdatascience.com/why-hasnt-reinforcement-learning-conquered-the-world-yet-459ae99982c6?source=rss----7f60cf5620c9---4)

