---
weight: 14
series: ["Programming: Learning Object-Oriented Concepts"]
series_order: 14
title: "The Four Pillars: Polymorphism"
authors:
  - jnonino
description: >
  Polymorphism is a core concept in object-oriented programming that allows objects of different classes to be treated as objects of a common base class. The term "polymorphism" comes from Greek, meaning "many forms". In OOP, it refers to the ability of a single interface to represent different underlying forms (data types or classes).
date: 2024-10-12
tags: ["Programming", "Object-Oriented Programming", "OOP", "The Four Pillars", "Polymorphism"]
---

Polymorphism enables writing flexible and reusable code by allowing us to work with objects at a more abstract level, without needing to know their specific types.

There are two main types of polymorphism in object-oriented programming:

1. **Compile-time polymorphism (Static polymorphism)**
   - Achieved through method *overloading*.
   - Resolved at compile time.

2. **Runtime polymorphism (Dynamic polymorphism)**
   - Achieved through method *overriding*.
   - Resolved at runtime.

Python primarily supports runtime polymorphism, as it is a dynamically typed language. However, we can demonstrate concepts similar to compile-time polymorphism as well.

Let's explore different aspects of polymorphism in Python:

### Duck typing

Python uses duck typing, which is a form of polymorphism. The idea is: "If it walks like a duck and quacks like a duck, then it must be a duck." In other words, Python cares more about the methods an object has than the type of the object itself.

```python
class Duck:
    def speak(self):
        return "Quack quack!"

class Dog:
    def speak(self):
        return "Woof woof!"

class Cat:
    def speak(self):
        return "Meow meow!"

def animal_sound(animal):
    return animal.speak()

# Usage
duck = Duck()
dog = Dog()
cat = Cat()

print(animal_sound(duck))  # Output: Quack quack!
print(animal_sound(dog))   # Output: Woof woof!
print(animal_sound(cat))   # Output: Meow meow!
```

In this example, `animal_sound()` works with any object that has a `speak()` method, regardless of its class.

### Method overriding

Method overriding is a key aspect of runtime polymorphism. It occurs when a derived class defines a method with the same name as a method in its base class.

```python
class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

# Usage
shapes = [Rectangle(5, 4), Circle(3)]

for shape in shapes:
    print(f"Area: {shape.area()}")

# Output:
# Area: 20
# Area: 28.27431
```

Here, `Rectangle` and `Circle` both override the `area()` method of the `Shape` class.

### Operator overloading

Python allows operator overloading, which is a form of compile-time polymorphism. It allows the same operator to have different meanings depending on the operands.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# Usage
v1 = Vector(2, 3)
v2 = Vector(3, 4)
v3 = v1 + v2

print(v3)  # Output: Vector(5, 7)
```

Here, we've overloaded the `+` operator for our `Vector` class.

### Abstract base classes

Python's `abc` module provides infrastructure for defining abstract base classes, which are a powerful way to define interfaces in Python.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

# Usage
def animal_sound(animal):
    return animal.make_sound()

dog = Dog()
cat = Cat()

print(animal_sound(dog))  # Output: Woof!
print(animal_sound(cat))  # Output: Meow!

# This will raise a TypeError
# animal = Animal()
```

Abstract base classes cannot be instantiated and force derived classes to implement certain methods, ensuring a consistent interface.

### Real-world Applications <a name="polymorphism-applications"></a>

Polymorphism is widely used in real-world applications:

1. **GUI frameworks**: Different widgets (buttons, text boxes) can respond to common events (click, hover) in their own ways.
2. **Database interfaces**: Different database systems can implement a common interface for querying, allowing applications to work with various databases without changing code.
3. **Plugin systems**: Applications can work with plugins through a common interface, regardless of the specific implementation of each plugin.
4. **Game development**: Different game entities can share common behaviors (move, collide) but implement them differently.

Here's a simple example of a plugin system:

```python
class Plugin(ABC):
    @abstractmethod
    def process(self, data):
        pass

class UppercasePlugin(Plugin):
    def process(self, data):
        return data.upper()

class ReversePlugin(Plugin):
    def process(self, data):
        return data[::-1]

class Application:
    def __init__(self):
        self.plugins = []

    def add_plugin(self, plugin):
        self.plugins.append(plugin)

    def process_data(self, data):
        for plugin in self.plugins:
            data = plugin.process(data)
        return data

# Usage
app = Application()
app.add_plugin(UppercasePlugin())
app.add_plugin(ReversePlugin())

result = app.process_data("Hello, World!")
print(result)  # Output: !DLROW ,OLLEH
```

This example demonstrates how polymorphism allows the `Application` class to work with different plugins through a common interface.

---

## References

1. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
2. Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design. Prentice Hall.
3. Phillips, D. (2010). Python 3 Object Oriented Programming. Packt Publishing.
4. Lutz, M. (2013). Learning Python: Powerful Object-Oriented Programming. O'Reilly Media.
5. Ramalho, L. (2015). Fluent Python: Clear, Concise, and Effective Programming. O'Reilly Media.
6. Van Rossum, G., Warsaw, B., & Coghlan, N. (2001). PEP 8 -- Style Guide for Python Code. Python.org. https://www.python.org/dev/peps/pep-0008/
7. Python Software Foundation. (n.d.). The Python Standard Library. Python.org. https://docs.python.org/3/library/

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
