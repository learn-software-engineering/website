---
weight: 4
series: ["Programming Starting Concepts"]
series_order: 4
title: "Functions"
authors:
  - jnonino
description: >
  In the vast and exciting world of programming, there are concepts that are fundamental pillars for any developer, regardless of their experience level. One of these concepts is functions. What are they? Why are they so crucial? Let's find out!
date: 2023-09-29
tags: ["Programming", "Starting Concepts", "Functions", "Methods"]
---

## What are functions?

A function, in simple terms, is a block of code that executes only when called. You can think of it as a small program within your main program, designed to perform a specific task. A function can also be seen as a black box: we pass an input (parameters), some internal processing occurs, and it produces an output (return value).

Functions allow us to segment our code into logical parts where each part performs a single action. This provides several benefits:

- **Reusability**: Once defined, we can execute (call) that code from anywhere in our program as many times as needed.
- **Organization**: It allows dividing a large program into smaller, more manageable parts.
- **Encapsulation**: Functions reduce complexity by hiding internal implementation details.
- **Maintainability**: If we need to make changes, we only have to modify the code in one place (the function) instead of tracking down all instances of that code.

**Procedures vs. Functions**

It is vital to distinguish between these two concepts. While a function always returns a value, a procedure performs a task but does not return anything. In some languages, this difference is clearer than in others. Python, for example, has functions that can optionally return values.

---

## Anatomy of a function

In Python, a function is declared using the `def` keyword, followed by the function name and parentheses. The code inside the function is called the *body* of the function and contains the set of instructions to execute to perform its task.

```python
def my_function():
    print("Hello from my function!")
```

To **call** or **invoke** a function, we simply use its name followed by parentheses:

```python
my_function() # Output: Hello from my function!
```

### Parameters and arguments

Functions become even more powerful when we pass information to them, known as **parameters**. These act as "*variables*" inside the function, allowing the function to work with different data each time it is called.

While **parameters** are variables defined in the function definition, **arguments** are the actual values passed when calling the function.

```python
def greet(name):
    print(f"Hello {name}!")

greet("Maria")
# Output:
#   Hello Maria!
```

Python allows **default parameters**, which have a predetermined value, making passing those arguments optional when calling the function. It also allows **named parameters** which enable passing arguments in any order by specifying their name.

```python
def greet(name="Maria", repetitions=3):
    repetition = 1
    while repetition <= repetitions:
        print(f"Hello {name}!")
        repetition += 1

greet()
# Output:
#   Hello Maria!
#   Hello Maria!
#   Hello Maria!

greet("Florencia", 4)
# Output:
#   Hello Florencia!
#   Hello Florencia!
#   Hello Florencia!
#   Hello Florencia!

greet(repetitions=2, name="Julian")
# Output
#   Hello Julian!
#   Hello Julian!
```

### Returning values

Functions can return a result or return value using the `return` keyword.

```python
def circle_area(radius):
    return 3.14 * (radius ** 2)

result = circle_area(10)
print(result) # Output: 314
```

The return value is passed back to where the function was called and can be assigned to a variable for later use.

Functions can also perform some task without explicitly returning anything. In Python this is known as returning `None`.

---

## Local and global variables

**Local variables** are defined inside a function and only exist in that scope, while **global variables** are defined outside and can be accessed from anywhere in the code. It is crucial to understand their **scope** (where a variable is accessible) and **lifetime** (how long a variable lives).

```python
x = 10 # x is global

def add():
    y = 5 # y is local
    return x + y

add() # Output: 15
print(y) # Error, y does not exist outside the function
```

We can read global variables from a function, but if we need to modify it we must declare it `global`.

```python
x = 10

def add():
    global x
    x = x + 5

add()
print(x) # 15
```

{{% alert color="warning" %}}
Ideally functions should only work with local variables and parameters. Limit the use of global variables. While they can be useful, they can also make code difficult to read and maintain.
{{% /alert %}}

---

## Best Practices

When creating functions we should follow certain principles and patterns:

- The name of a function should clearly indicate its purpose.
- Make functions small, simple, and focused on one task. A function should do one thing and do it well.
- Use descriptive names for functions and parameters.
- Avoid side effects and modifying global variables.
- Properly document the purpose and usage of each function.
- Limit the number of parameters, ideally 0 to 3 parameters.

Following these best practices will help us create reusable, encapsulated, and maintainable functions.

---

## Conclusion

Functions are core components in programming, allowing us to organize, reuse, and encapsulate code. By defining functions that perform a single task we keep our programs simplified, easy to understand, and modify. By understanding and mastering this concept, you not only improve the quality of your code but also your efficiency as a developer.

---

## References

1. McConnell, S. (2004). *Code Complete*. Microsoft Press.
2. Joyanes Aguilar, L. (2008). *Fundamentos de programación: algoritmos, estructura de datos y objetos*. McGraw-Hill.
3. Kindler, E., & Krivy, I. (2011). *Object-Oriented Simulation of systems with Java: A working introduction*. BoD–Books on Demand.
4. Python Software Foundation. (2022). *Python Official Documentation*.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
