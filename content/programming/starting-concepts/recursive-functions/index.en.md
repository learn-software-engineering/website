---
weight: 5
series: ["Programming: Learning the Starting Concepts"]
series_order: 5
title: "Recursive Functions"
authors:
  - jnonino
description: >
  Recursion is a fundamental concept in programming that allows a function to call itself. At first it may seem counter-intuitive, but mastering this approach opens the door to elegant solutions for certain problems.
date: 2023-09-30
tags: ["Programming", "Starting Concepts", "Functions", "Methods", "Recursion"]
---

## Recursion: the art of calling yourself

Imagine a box of mirrors where each mirror reflects what it sees in the next, creating an infinite series of reflections. Recursion in programming is something similar. It is a technique where a function calls itself directly or indirectly. This creates a cycle where the function solves a problem by dividing it into smaller instances of the same problem, until reaching a simple base case that can be solved directly.

For example, let's imagine a function that prints a countdown:

```python
def countdown(number):

    if number > 0:
        print(number)
        countdown(number - 1)
    else:
        print("Blastoff!")

countdown(5)
```

This function calls itself recursively decrementing the number each time until reaching 0, and then prints the blastoff message.

Recursion is a declarative approach that focuses on dividing a problem into recursive cases without needing to explicitly control the loop using iterators or counters like in imperative programming.

---

## The structure of a recursive function

The power of recursion lies in its simplicity. However, it is essential to understand its structure to avoid common pitfalls. A typical recursive function has two main parts:

1. **Base case**: The simplest case with a known solution that doesn't require recursion. It is the stopping condition that halts the recursion. Without a base case, we would fall into infinite recursion which eventually overflows the call stack.
2. **Recursive case**: This is where the magical recursive call occurs. At this point, the function calls itself with a modified argument, usually a reduced version of the original problem.

---

## Classic recursion examples

### Factorial

The factorial of a positive integer \\(n\\) is the product of all positive integers less than or equal to \\(n\\). For example:

- \\(5! = 5 * 4 * 3 * 2 * 1 = 120\\)
- \\(4! = 4 * 3 * 2 * 1 = 24\\)
- \\(3! = 3 * 2 * 1 = 6\\)

Here is the Python code for calculating factorial using recursion:

```python
def factorial(n):
    if n == 1:
        return 1       # Base case
    return n * factorial(n-1) # Recursive case
```

- **Base case**: The base case is the simplest, smallest instance of the problem that can be answered directly. For factorial, when \\(n = 1\\), the result is \\(1\\).
- **Recursive case**: If \\(n\\) is greater than \\(1\\), the function calls itself with \\(n-1\\), and multiplies the result by \\(n\\).

Let's say you want to calculate the factorial of \\(5\\), so you call `factorial(5)`.

Here is what happens:

1. **Step 1**: Since \\(n = 5\\) is not \\(1\\), the function calls `factorial(4)`, then multiplies the result by \\(5\\).
2. **Step 2**: Now, inside `factorial(4)`, \\(n = 4\\), so the function calls `factorial(3)`, then multiplies the result by \\(4\\).
3. **Step 3**: Inside `factorial(3)`, \\(n = 3\\), so it calls `factorial(2)`, then multiplies the result by \\(3\\).
4. **Step 4**: Inside `factorial(2)`, \\(n = 2\\), so it calls `factorial(1)`, then multiplies the result by \\(2\\).
5. **Step 5**: Finally, `factorial(1)` reaches the base case, where \\(n = 1\\), so it returns \\(1\\).

Now the results unwind:

- `factorial(2)` returns \\(2 * 1 = 2\\)
- `factorial(3)` returns \\(3 * 2 = 6\\)
- `factorial(4)` returns \\(4 * 6 = 24\\)
- `factorial(5)` returns \\(5 * 24 = 120\\)

The final result is \\(120\\), which is the value of \\(5!\\).

Here is a visual representation of the call stack:

```python
factorial(5)
  -> factorial(4)
    -> factorial(3)
      -> factorial(2)
        -> factorial(1)
          return 1
        return 2
      return 6
    return 24
  return 120
```

### Fibonacci sequence

The Fibonacci sequence is a series of numbers where each number is the sum of the previous two. It starts with \\(0\\) and \\(1\\), and each subsequent number is the sum of the two numbers before it. The beginning of the sequence is: \\(0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\\)

Here is the Python code for calculating the \\(n^th\\) Fibonacci number using tail recursion:

```python
def fibonacci(n, a=0, b=1):
    if n == 0:
        return a
    return fibonacci(n-1, b, a+b)
```

The function takes three parameters:

- \\(n\\): The position of the desired number in the sequence.
- \\(a\\) and \\(b\\): Two numbers that aid in the sequence calculation.

Here is a breakdown of how the function works:

1. **Base case**: If \\(n\\) is \\(0\\), the function returns \\(a\\). This is the value of the \\(n^th\\) number in the sequence.
2. **Recursive case**: If \\(n\\) is not \\(0\\), the function calls itself with \\(n-1\\), \\(b\\), and \\(a+b\\). These parameters change the position in the sequence and prepare the next numbers for summation.

Suppose we want to find the \\(5^th\\) number in the Fibonacci sequence by calling `fibonacci(5)`.

Here is what happens:

1. **Step 1**: Since \\(n = 5\\), it calls `fibonacci(4, 1, 1)` (because \\(a = 0\\), \\(b = 1\\), \\(a + b = 1\\)).
2. **Step 2**: Since \\(n = 4\\), it calls `fibonacci(3, 1, 2)` (because \\(a = 1\\), \\(b = 1\\), \\(a + b = 2\\)).
3. **Step 3**: Since \\(n = 3\\), it calls `fibonacci(2, 2, 3)` (because \\(a = 1\\), \\(b = 2\\), \\(a + b = 3\\)).
4. **Step 4**: Since \\(n = 2\\), it calls `fibonacci(1, 3, 5)` (because \\(a = 2\\), \\(b = 3\\), \\(a + b = 5\\)).
5. **Step 5**: Since \\(n = 1\\), it calls `fibonacci(0, 5, 8)` (because \\(a = 3\\), \\(b = 5\\), \\(a + b = 8\\)).
6. **Step 6**: Since \\(n = 0\\), it returns \\(a\\), which is \\(5\\).

The result is \\(5\\), which is the \\(5^th\\) number in the Fibonacci sequence.

Here is a visual representation of the call stack:

```python
fibonacci(5, 0, 1)
  -> fibonacci(4, 1, 1)
    -> fibonacci(3, 1, 2)
      -> fibonacci(2, 2, 3)
        -> fibonacci(1, 3, 5)
          -> fibonacci(0, 5, 8)
            return 5
```

---

## Advantages and Disadvantages

Recursion has certain advantages:

- It can result in simple, elegant solutions for problems that easily break down into subproblems.
- It eliminates the need for explicit loop control.
- It mirrors the mathematical structure of a recursive definition.

The disadvantages include:

- It can be less efficient (high memory consumption) than iteration due to repeated function calls and stack frame creation.
- Too much recursion can overflow the call stack and cause crashes.
- It can be harder to debug and analyze than iteration.

Therefore, recursion is a powerful tool that should be used judiciously in appropriate cases.

---

## Recursion vs Iteration

Recursion and iteration (using loops) are parallel tools and we can use either one to solve many problems. Both techniques have the potential to solve the same problems, but their implementation and efficiency may vary. Let's take the factorial example:

**Iterative**

```python
def factorial_iterative(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
```

**Recursive**

```python
def factorial_recursive(n):
    if n == 1:
        return 1
    return n * factorial(n-1)
```

The iterative version is more efficient in terms of space, but the recursive is cleaner and easier to understand. The choice between recursion and iteration often depends on the specific problem, memory constraints, and programmer preferences.

---

## Conclusion

Recursion is a key technique that allows writing elegant, natural, and efficient algorithms when properly leveraged. Understanding how to break down a problem into recursive cases is essential to master this skill. Recursion provides a unique declarative alternative to solve complex problems without managing explicit loops. However, it is crucial to always remember to define an adequate base case and be aware of recursion limitations in terms of efficiency and memory usage. Knowing how to combine recursion and iteration gives flexibility when creating optimal solutions.

As always, the key lies in finding the right balance and using the right tool for the right job.

---

## References

1. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2009). Introduction to Algorithms. MIT Press.
2. Kindler, E., & Krivy, I. (2011). *Object-Oriented Simulation of systems with Java: A working introduction*. BoD–Books on Demand.
3. Lutz, M. (2013). *Learning Python: Powerful Object-Oriented Programming*. O'Reilly Media, Incorporated.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
