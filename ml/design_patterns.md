# Design Patterns

<!-- MarkdownTOC -->

- What is a Design Pattern?
- How to Classify Design Patterns?
- Design Pattern Cheatsheet
- References

<!-- /MarkdownTOC -->

## What is a Design Pattern?

A _design pattern_ is defined by four essential elements:

**Name:** A way of defining the general context and vocabulary of how and what the DP is used to solve.

**Problem:** Describes when to apply the pattern.

**Solution:** The way classes, interfaces and objects are designed to respond to a problem.

**Consequences:** The trade-offs we have to consider once we decide to apply a given DP.

## How to Classify Design Patterns?

Design patterns are categorized according to the type of reuse:

**Behavioral:** How responsibility is sharedand information is propagated through different objects.

**Structural:** Concerns about the interaction processes between the different objects and classes.

**Creational:** Allows decoupling and optimizing the creation steps of different objects.


1. Strategy

Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

2. Mediator

Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently.

3. State

Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.

4. Builder

Separate the construction of a complex object from its representation so that the same construction process can create different representations.

5. Prototype

Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

6. Adapter

Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn’t otherwise because of incompatible interfaces.

7. Decorator

Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.



## Design Pattern Cheatsheet

- Design Patterns for Machine Learning
- More Design Patterns for Machine Learning
- Data Mesh Observability Pattern

- Design Patterns in Machine Learning for MLOps
- How (not) to do MLOps

- Common Patterns for Serving ML Models in Production


## References

Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides. Design Patterns: Elements of Reusable Object-Oriented Software. ISBN: 978-0201633610. 1994.

[Design Patterns for Machine Learning](https://towardsdatascience.com/design-patterns-for-machine-learning-410be845c0db)

[Understand Machine Learning Through 7 Software Design Patterns](https://betterprogramming.pub/machine-learning-through-7-design-patterns-35a8d5844cf6)

[Understand Machine Learning through More Design Patterns](https://towardsdatascience.com/understand-machine-learning-through-more-design-patterns-9c8430fd2ae8)

[Data Mesh Observability Pattern](https://towardsdatascience.com/data-mesh-observability-pattern-467438627572)


[Design Patterns in Machine Learning for MLOps](https://towardsdatascience.com/design-patterns-in-machine-learning-for-mlops-a3f63f745ce4)

[How (not) to do MLOps](https://towardsdatascience.com/how-not-to-do-mlops-96244a21c35e)


[Serving ML Models in Production: Common Patterns](https://www.kdnuggets.com/2021/10/serving-ml-models-production-common-patterns.html)