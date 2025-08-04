---
weight: 15
series: ["Object-Oriented Programming"]
series_order: 15
title: "The Four Pillars: Abstraction"
authors:
  - jnonino
description: >
  Abstraction is the process of hiding the complex implementation details and showing only the necessary features of an object. It's about creating a simplified view of an object that represents its essential characteristics without including background details or explanations.
date: 2024-10-12
tags: ["Programming", "Object-Oriented Programming", "OOP", "The Four Pillars", "Abstraction"]
---

Key aspects of abstraction include:

1. **Simplification**: Abstraction reduces complexity by hiding unnecessary details.
2. **Focusing on essential features**: It emphasises what an object does rather than how it does it.
3. **Separation of concerns**: It allows separating the interface of a class from its implementation.
4. **Modularity**: Abstraction promotes modular design by defining clear boundaries between components.

### Abstract classes and interfaces

In many object-oriented languages, abstraction is implemented through abstract classes and interfaces. While Python doesn't have a built-in interface concept, we can achieve similar functionality using abstract base classes. Python's `abc` module provides infrastructure for defining abstract base classes:

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Usage
# shapes = [Shape()]  # This would raise TypeError
shapes = [Rectangle(5, 4), Circle(3)]

for shape in shapes:
    print(f"Area: {shape.area()}, Perimeter: {shape.perimeter()}")

# Output:
# Area: 20, Perimeter: 18
# Area: 28.27431, Perimeter: 18.84954
```

In this example:
- `Shape` is an abstract base class that defines the interface for all shapes.
- `Rectangle` and `Circle` are concrete classes that implement the `Shape` interface.
- We can't instantiate `Shape` directly, but we can use it as a common type for all shapes.

### Implementing abstraction in Python

While abstract base classes provide a formal way to define interfaces in Python, abstraction can also be achieved through convention and documentation. Here's an example of abstraction without using `ABC`:

```python
class Database:
    def connect(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def execute(self, query):
        raise NotImplementedError("Subclass must implement abstract method")

class MySQLDatabase(Database):
    def connect(self):
        print("Connecting to MySQL database...")

    def execute(self, query):
        print(f"Executing MySQL query: {query}")

class PostgreSQLDatabase(Database):
    def connect(self):
        print("Connecting to PostgreSQL database...")

    def execute(self, query):
        print(f"Executing PostgreSQL query: {query}")

def perform_database_operation(database):
    database.connect()
    database.execute("SELECT * FROM users")

# Usage
mysql_db = MySQLDatabase()
postgres_db = PostgreSQLDatabase()

perform_database_operation(mysql_db)
perform_database_operation(postgres_db)

# Output:
# Connecting to MySQL database...
# Executing MySQL query: SELECT * FROM users
# Connecting to PostgreSQL database...
# Executing PostgreSQL query: SELECT * FROM users
```

In this example:
- `Database` is an abstract base class (though not using `ABC`) that defines the interface for all database types.
- `MySQLDatabase` and `PostgreSQLDatabase` are concrete implementations.
- `perform_database_operation` works with any object that adheres to the `Database` interface.

### Design principles and patterns

Abstraction is a key component of several important design principles and patterns:

1. **SOLID Principles**:
   - Single Responsibility Principle (SRP).
   - Open/Closed Principle (OCP).
   - Liskov Substitution Principle (LSP).
   - Interface Segregation Principle (ISP).
   - Dependency Inversion Principle (DIP).

2. **Design Patterns**:
   - Factory method pattern.
   - Abstract factory pattern.
   - Strategy pattern.
   - Template method pattern.

Let's implement the Strategy Pattern as an example:

```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        print("Performing bubble sort")
        return sorted(data)  # Using Python's built-in sort for simplicity

class QuickSort(SortStrategy):
    def sort(self, data):
        print("Performing quick sort")
        return sorted(data)  # Using Python's built-in sort for simplicity

class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy

    def sort(self, data):
        return self.strategy.sort(data)

# Usage
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

bubble_sorter = Sorter(BubbleSort())
print(bubble_sorter.sort(data))

quick_sorter = Sorter(QuickSort())
print(quick_sorter.sort(data))

# Output:
# Performing bubble sort
# [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
# Performing quick sort
# [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

This Strategy Pattern example demonstrates how abstraction allows us to define a family of algorithms, encapsulate each one, and make them interchangeable. The `Sorter` class doesn't need to know the details of how each sorting algorithm works; it just knows that it can call the `sort` method on any `SortStrategy` object.

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
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}

---
