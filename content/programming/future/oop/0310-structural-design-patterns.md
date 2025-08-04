---
draft: true
title: "Structural Design Patterns"
date: 2023-10-20
tags: ["OOP", "Design Patters", "UML"]
# images:
#   - /images/banners/content/programming/0300-oop/0307-structural-design-patterns.png
---





## Best Practices and Design Patterns

### The Liskov Substitution Principle (LSP)

The LSP, one of the SOLID principles, is crucial for effective use of polymorphism. It states that objects of a superclass should be replaceable with objects of its subclasses without affecting the correctness of the program.

Example of an incorrect implementation, it violates LSP:

```python
class Bird(ABC):
    @abstractmethod
    def fly(self):
        pass

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins can't fly")
```

A better design to avoid violating LSP could be:

```python
class Bird(ABC):
    @abstractmethod
    def move(self):
        pass

class FlyingBird(Bird):
    def move(self):
        return "Flying through the air"

class SwimmingBird(Bird):
    def move(self):
        return "Swimming in the water"

class Penguin(SwimmingBird):
    pass

class Sparrow(FlyingBird):
    pass
```

### The Liskov Substitution Principle (LSP)

Barbara Liskov's principle states that objects of a superclass should be replaceable with objects of its subclasses without affecting the correctness of the program. This is crucial for maintaining system integrity and preventing unexpected behaviors.

Example of violating LSP:
```python
class Rectangle:
    def set_width(self, width):
        self._width = width

    def set_height(self, height):
        self._height = height

    def area(self):
        return self._width * self._height

class Square(Rectangle):  # Violates LSP
    def set_width(self, width):
        self._width = width
        self._height = width  # Forces square behaviour

    def set_height(self, height):
        self._height = height
        self._width = height  # Forces square behaviour
```

### Strategy Pattern

The Strategy Pattern is a classic example of using polymorphism to implement different algorithms:

```python
from abc import ABC, abstractmethod
from typing import List

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data

class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self.strategy = strategy

    def sort(self, data: List[int]) -> List[int]:
        return self.strategy.sort(data.copy())
```

We will dive deeper on design patters later in the course.
