---
weight: 5
series: ["Data Structures"]
series_order: 5
title: "Stacks"
authors:
  - jnonino
description: >
  Stacks are an abstract data structure that operates under the LIFO (last in, first out) principle, where the last element to enter is the first to leave.
date: 2023-11-02
tags: ["Programming", "Data Structures", "Lists", "Linked Lists", "Stacks"]
---

The **LIFO** nature of stacks is due to the fact that only the top element can be accessed and manipulated. The operation of placing an element on the stack is known as *"push"*, while removing an element from the stack is a *"pop"*. The LIFO operation causes the last element placed in a stack to be the first to be removed.

{{< figure
    src="stacks.jpg"
    alt="Diagram of a stack"
    caption="Diagram of a stack"
    >}}

---

## Main operations

The primary operations supported by a stack structure are:

- **Push:** adds an element to the top of the stack.
- **Pop:** removes the element at the top of the stack.
- **Peek:** allows accessing the top element without removing it from the stack.
- **isEmpty:** checks if the stack is empty.

Most languages like Python and Java provide stack implementations in their standard libraries.

---

## Implementation

A stack can be implemented using a linked list so that each node points to the previous node.

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/main/programming/data_structures/stacks.py"
    type="python"
    >}}

---

## Usage examples

Stacks have many uses in programming:

- **Execution stack (call stack)**: records pending function calls to resolve. Implements expected LIFO behaviour.

- **Browser stack**: allows going back (undo) in the browser history similarly to a LIFO stack.

- **Math expression execution**: stacks can verify parentheses, brackets, braces, etc.

- **Algorithms and data structures**: like in the quicksort algorithm and in data path implementations.

---

## Conclusion

Stacks are versatile data structures thanks to their LIFO operation principle. Having a good command of stacks, their uses and applications is essential in computer science.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}
