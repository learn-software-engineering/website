---
weight: 1
series: ["Data Structures"]
series_order: 1
title: "Introduction"
authors:
  - jnonino
description: >
  Data structures are ways of organizing and storing information in a computer program so that it can be accessed and modified efficiently. As programmers, it is essential to understand the different data structures available and know when to apply each one to optimize the performance and utility of our programs.
date: 2025-08-04
tags: ["Programming", "Data Structures"]
---

Data structures are ways of organizing and storing information in a computer program so that it can be accessed and modified efficiently. As programmers, it is essential to understand the different data structures available and know when to apply each one to optimize the performance and utility of our programs.

A data structure is a particular way of organizing data in the computer's memory so that it can be used efficiently. Data structures come in many forms, such as arrays, lists, stacks, queues, graphs, trees, hashes, etc.

Each structure organizes the data according to a specific logical model and supports efficient operations to access, modify, add, and delete elements according to that model. For example, an array organizes elements sequentially in memory to facilitate random access by indices. A linked list connects elements in memory using "nodes" with references to the next node to facilitate insertion and deletion.

By choosing the appropriate data structure for the task to be solved, we can write more efficient programs and reduce computational complexity, using fewer resources like memory and processing.

Data structures help us:

- Organize large amounts of data to make them easier to access and modify.

- Model complex relationships between data, such as with graphs and trees.

- Access and modify data efficiently, optimizing performance.

- Reuse existing code and data structures instead of having to rewrite solutions from scratch.

For example, a program that needs to store thousands of user records benefits from using a hash data structure to associate each user with data like first name, last name, email, etc. This way specific users can be found very quickly without having to iterate over the entire collection.

Another example are binary search trees, which allow finding elements very quickly in ordered sets of millions of elements. This is achieved by discarding halves of the tree as the desired element is searched.

---

## Types of data structures

There are many types of data structures. Below are some useful categories to classify them.

### According to relationship between elements

- **Linear**: Elements are organized sequentially one after the other. For example, arrays, lists, stacks, queues.

- **Non-linear**: Elements are organized in a hierarchy or graph. This is the case with trees and graphs.

### According to element type

- **Homogeneous**: Store a single data type. For example, arrays in a language like Java.

- **Heterogeneous**: Allow different data types. Objects, records are examples of this classification.

### According to access mode

- **Sequential access**: Elements can only be accessed in sequential order. For example, linked lists.

- **Direct access**: Any element can be accessed directly by its position. Arrays fall into this group.

- **Associative access**: Elements are accessed by an associated key. Dictionaries, hashes fall into this category.

### According to functionality

- **Arrays**: Fast access to elements by index but difficult to insert/delete.

- **Lists**: Easy to insert/delete anywhere but sequential access is slow.

- **Stacks**: LIFO (Last In First Out) access. Useful for undo/redo.

- **Queues**: FIFO (First In First Out) access. Useful for event processing.

- **Trees**: Allow modeling hierarchical relationships like file directories or task dependencies.

- **Graphs**: Allow modeling interconnectivity networks like maps, social relationships, etc.

- **Hashes / Dictionaries**: Associate elements with unique keys for ultra fast access.

This classification is not exhaustive but gives an idea of the diversity of data structures and their different properties that allow us to efficiently model complex problems.

---

## Example

Let's look at a Python example to see how a data structure is created and used. Suppose we want to represent a print queue where prints are processed in order of arrival (FIFO).

First, we define a `PrintQueue` class to represent our queue:

```python
class PrintQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.pop(0)

    def is_empty(self):
        return len(self.items) == 0
```

Then we use it to add prints and process them in order:

```python
print_queue = PrintQueue()

print_queue.enqueue("document1.doc")
print_queue.enqueue("image3.jpg")
print_queue.enqueue("presentation.ppt")

while not print_queue.is_empty():
    next_item = print_queue.dequeue()
    print(f"Printing {next_item}")
```

This will print:

```
Printing document1.doc
Printing image3.jpg
Printing presentation.ppt
```

With a data structure like the queue we implement FIFO logic in a simple, reusable way. And this is just one example, the possibilities are endless!

---

## Conclusion

Data structures are fundamental programming tools that allow us to optimally organize information to solve complex problems. Knowing the different types of structures available, like arrays, lists, stacks, queues, hashes, graphs, and trees, allows us to build more efficient programs. I hope this introduction has given you some knowledge and tools to start mastering this exciting topic!
