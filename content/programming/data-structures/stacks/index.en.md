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

```python
class Node:
  def __init__(self, value):
    self.value = value
    self.previous = None


class Stack:
  def __init__(self):
    self.top = None
    self.size = 0

  def push(self, value):
    new_node = Node(value)
    if self.top is None:
      self.top = new_node
    else:
      new_node.previous = self.top
      self.top = new_node
    self.size += 1

  def pop(self):
    if self.top is None:
      return None
    top_node = self.top
    self.top = self.top.previous
    self.size -= 1
    return top_node.value

  def peek(self):
    if self.top is None:
      return None
    return self.top.value

  def is_empty(self):
    return self.top is None  # Returns true if top is None

  def __len__(self):
    return self.size

  def __str__(self):
    values = []
    current = self.top
    while current:
      values.append(str(current.value))
      current = current.previous
    return "\n".join(values)
```

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
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}
