---
draft: true
weight: 1101
title: "Introduction"
description: >
  Design patterns are tried-and-true solutions to common problems that arise in software development. They represent best practices and are used to create organized, clean, and scalable code. This article covers various design patterns with examples in Python."
date: 2023-03-27
tags: ["Programming", "Design Patters", "Singleton", "Adapter", "Observer"]
---

## Types of design patterns

- **Creational patterns**: are focused on the process of object creation, abstracting the instantiation process.

- **Structural patterns**: are concerned with the composition of classes or objects, simplifying the structure and identifying relationships between objects.

- **behavioural patterns**: define ways for objects to communicate and interact, standardizing how objects cooperate.

## Examples of design patterns

Below are examples of three common design patterns implemented in Python.

### Singleton pattern

The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance.

```python
class Singleton:
    _instance = None

    @staticmethod
    def getInstance():
        if Singleton._instance == None:
            Singleton()
        return Singleton._instance

    def __init__(self):
        if Singleton._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Singleton._instance = self
```

### Adapter pattern

The Adapter pattern allows incompatible interfaces to work together, wrapping one class with another that has the expected interface.

```python
class OldSystem:
    def old_method(self):
        return "Old system method"

class Adapter:
    def __init__(self, old_system):
        self.old_system = old_system

    def new_method(self):
        return self.old_system.old_method()
```

### Observer pattern

The Observer pattern defines a one-to-many dependency between objects, allowing multiple observers to respond to changes in a subject.

```python
class Subject:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update(self)

class Observer:
    def update(self, subject):
        pass
```

## Conclusion

Design patterns are essential tools that guide software developers in creating efficient, maintainable, and scalable code. By understanding and applying these patterns, developers can avoid common pitfalls and build robust software systems.

Further reading and exploration of design patterns in various programming languages are highly recommended. A classic reference on this topic is the book "Design Patterns: Elements of Reusable Object-Oriented Software" by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides.
