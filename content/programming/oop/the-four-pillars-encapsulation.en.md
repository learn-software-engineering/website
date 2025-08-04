---
weight: 12
series: ["Object-Oriented Programming"]
series_order: 12
title: "The Four Pillars: Encapsulation"
authors:
  - jnonino
description: >
  Encapsulation is often described as the first pillar of object-oriented programming. It is the mechanism of bundling the data (attributes) and the methods (functions) that operate on the data within a single unit or object. This concept is also often referred to as data hiding because the object's internal representation is hidden from the outside world.
date: 2024-10-12
tags: ["Programming", "Object-Oriented Programming", "OOP", "The Four Pillars", "Encapsulation"]
---

The importance of encapsulation lies in several key aspects:

1. **Data protection**: By controlling access to object data through methods, we can ensure that the data remains consistent and valid.
2. **Modularity**: Encapsulation allows objects to be self-contained, making it easier to understand and maintain code.
3. **Flexibility**: The internal implementation can be changed without affecting other parts of the code that use the object.
4. **Reduced complexity**: By hiding the details of internal workings, encapsulation reduces the complexity of the overall system from an external perspective.

### Implementation in Python

Python provides several mechanisms to implement encapsulation. Let's explore these with examples:

#### 1. Using private attributes

In Python, we can create private attributes by prefixing the attribute name with double underscores (`__`). This triggers name mangling, which makes the attribute harder to access from outside the class.

```python
class BankAccount:
    def __init__(self, account_number, balance):
        self.__account_number = account_number  # Private attribute
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        return self.__balance

# Usage
account = BankAccount("1234567890", 1000)
print(account.get_balance())  # Output: 1000
account.deposit(500)
print(account.get_balance())  # Output: 1500
account.withdraw(200)
print(account.get_balance())  # Output: 1300

# This will raise an AttributeError
# print(account.__balance)
```

In this example:
- `__account_number` and `__balance` are private attributes.
- We provide public methods (`deposit`, `withdraw`, `get_balance`) to interact with these private attributes.
- Direct access to `__balance` from outside the class will raise an `AttributeError` exception.

#### 2. Using properties

Python's `@property` decorator allows us to define methods that can be accessed like attributes, providing a more Pythonic way of implementing getters and setters.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value > 0:
            self._radius = value
        else:
            raise ValueError("Radius must be positive")

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

# Usage
circle = Circle(5)
print(circle.radius)  # Output: 5
print(circle.area)    # Output: 78.53975

circle.radius = 7
print(circle.radius)  # Output: 7
print(circle.area)    # Output: 153.93791

# This will raise a ValueError
# circle.radius = -1
```

In this example:
- `_radius` is a protected attribute (single underscore is a convention for protected attributes in Python).
- The `radius` property provides get and set access to `_radius` with validation.
- The `area` property is read-only and calculated on-the-fly.

### Benefits and best practices

The benefits of encapsulation are numerous:

1. **Improved maintainability**: Changes to the internal implementation don't affect external code that uses the class.
2. **Enhanced security**: Private attributes can't be accidentally modified from outside the class.
3. **Flexibility in implementation**: You can change how data is stored or calculated without changing the public interface.
4. **Better abstraction**: Users of the class don't need to know about its internal workings.

Best practices for encapsulation in Python include:

- Use private attributes (double underscore prefix) for data that should not be accessed directly from outside the class.
- Provide public methods or properties for controlled access to internal data.
- Use properties instead of get/set methods for a more Pythonic approach.
- Document the public interface clearly, including any side effects of methods.

Let's look at a more complex example that demonstrates these practices:

```python
class Employee:
    def __init__(self, name, salary):
        self.__name = name
        self.__salary = salary
        self.__projects = []

    @property
    def name(self):
        return self.__name

    @property
    def salary(self):
        return self.__salary

    @salary.setter
    def salary(self, value):
        if value > 0:
            self.__salary = value
        else:
            raise ValueError("Salary must be positive")

    def add_project(self, project):
        """
        Add a project to the employee's project list.

        :param project: string representing the project name
        """
        self.__projects.append(project)

    def remove_project(self, project):
        """
        Remove a project from the employee's project list.

        :param project: string representing the project name
        :return: True if project was removed, False if not found
        """
        if project in self.__projects:
            self.__projects.remove(project)
            return True
        return False

    @property
    def project_count(self):
        return len(self.__projects)

    def __str__(self):
        return f"Employee: {self.__name}, Salary: ${self.__salary}, Projects: {self.project_count}"

# Usage
emp = Employee("John Doe", 50000)
print(emp.name)  # Output: John Doe
print(emp.salary)  # Output: 50000

emp.add_project("Project A")
emp.add_project("Project B")
print(emp.project_count)  # Output: 2

emp.salary = 55000
print(emp)  # Output: Employee: John Doe, Salary: $55000, Projects: 2

emp.remove_project("Project A")
print(emp.project_count)  # Output: 1

# This will raise an AttributeError
# print(emp.__projects)
```

This example demonstrates:
- Private attributes (`__name`, `__salary`, `__projects`)
- Properties for controlled access (`name`, `salary`, `project_count`)
- Public methods for manipulating private data (`add_project`, `remove_project`)
- Clear documentation of method behaviour
- A custom `__str__` method for a nice string representation of the object

By following these practices, we create a class that is both flexible and robust, embodying the principle of encapsulation.

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
