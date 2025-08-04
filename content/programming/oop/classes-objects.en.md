---
weight: 2
series: ["Object-Oriented Programming"]
series_order: 2
title: "Classes and objects"
authors:
  - jnonino
description: >
  In object-oriented programming, classes and objects are the key concepts to understand how we model elements of reality and define their structure and behaviour within software. Let's look in detail at the anatomy of a class, how to create objects from it to use their properties and methods, and other key details of their relationship.
date: 2023-10-02
tags: ["Programming", "Object-Oriented Programming", "OOP", "Classes", "Objects"]
---

## Anatomy of a class

A class acts as a **blueprint or mould** to construct similar objects, defining their common characteristics and functionalities. It is similar to the blueprint used to construct houses in the same neighbourhood: they all share certain key attributes.

The typical components of a class are:

**Attributes (properties)**: Variables that characterise the object. For example, for a `Person` class, attributes like `name`, `age`, `ID`, etc.

```python
class Person:
  id = ""
  name = ""
  age = 0
```

**Methods**: Functions that define behaviours. For example, a `Person` can `walk()`, `talk()`, `eat()`, etc. They access the attributes to implement said functionality.

**Constructor**: Special `__init__()` method that executes when instantiating the class and allows initialising attributes.

**Destructor**: `__del__()` method that executes when deleting the instance, freeing up resources. Optional in some languages.

---

## Creating objects

From the class we generate **objects**, which are specific **instances** with their own defined attributes. Let's say the House class is the blueprint, and a specific house on a particular street is the object.

In code, we create an object by invoking the class as if it were a method:

```python
# Person class
class Person:
  def __init__(self, n, a):
    self.name = n
    self.age = a

# Specific Person objects
john = Person("John", 30)
mary = Person("Mary", 35)
```

Each object shares the general structure and behaviour but can store different data.

### Using properties and methods

We now have a `Person` class and a `john` object of type `Person`. How do we interact with the object?

- **Properties**: It is possible to access the value of an object attribute using the object reference (`john`) and the attribute name.

```python
john.name  # "John"
john.age   # 30
```

- **Methods**: Are invoked in the same way as accessing attributes but adding parentheses, and inside them, the arguments that are passed if it takes any.

```python
# Person class
class Person:
  def __init__(self, n, a):
      self.name = n
      self.age = a

  def eat(self, food):
      print(f"Eating {food}")

# Specific Person object
john = Person("John", 30)
john.eat("pizza") # Prints "Eating pizza"
```

The john object now has its own **state** (properties) and **behaviour** (methods).

---

## Self vs This

An important detail in methods is how they access the object's attributes and other methods. Here another difference between languages comes into play:

- **Self**: In Python, attributes and methods are accessed within the class by prepending `self`. This points to the instantiated object.

```python
class Person:

  def __init__(self, name):
    self.name = name

  def greet(self):
    print(f"Hello! I'm {self.name}")

john = Person("John")
john.greet()
# Prints "Hello! I'm John"
```

- **This**: In Java or C#, `this` is used instead of self. It fulfils the same functionality of pointing to the object's members.

```java
public class Person {

  private String name;

  public Person(String name) {
    this.name = name;
  }

  public void greet() {
    System.out.println("Hello! I'm " + this.name);
  }
}

Person john = new Person("John");
john.greet();
// Prints "Hello! I'm John"
```

---

## Conclusion

Classes and objects are the key concepts in OOP, allowing modelling real-world entities and generating modular, generic components of our system to construct more robust and easy to understand programmes.

---

{{% alert color="info" %}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{% /alert %}}

---
