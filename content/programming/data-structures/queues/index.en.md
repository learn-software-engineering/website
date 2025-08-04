---
weight: 6
series: ["Data Structures"]
series_order: 6
title: "Queues"
authors:
  - jnonino
description: >
  Queues are an abstract data structure that operates under the FIFO (first in, first out) principle, where the first element to enter is also the first to leave. Queues are used to order elements so that the first to arrive is processed first. Understanding their operation is essential for any programmer.
date: 2023-11-03
tags: ["Programming", "Data Structures", "Lists", "Linked Lists", "Queues"]
---

The FIFO (first in, first out) nature of queues is because only the initial element can be accessed and manipulated. When an element is added to the queue it is known as *"enqueue"*, while removing an element is called *"dequeue"*.

This causes the first element to be added to the queue to also be the first to be removed, hence its FIFO behaviour.

{{< figure
    src="queues.jpg"
    alt="Diagram of a queue"
    caption="Diagram of a queue"
    >}}

---

## Main operations

The basic queue operations are:

- **Enqueue:** Adds an element to the end of the queue.
- **Dequeue:** Removes the element from the front of the queue.
- **Peek:** Gets the front element without removing it.
- **isEmpty:** Checks if the queue is empty.

---

## Implementation

Like stacks, queues can be implemented using linked lists.
Elements are added at the end and removed from the front by keeping references to both ends.

```python
class Node:
  def __init__(self, value):
    self.value = value
    self.next = None


class Queue:
  def __init__(self):
    self.front = None
    self.end = None
    self.size = 0

  def enqueue(self, value):
    new_node = Node(value)
    if self.end is None:
      self.end = new_node
      self.front = new_node
      return
    self.end.next = new_node
    self.end = new_node
    self.size += 1

  def dequeue(self):
    if self.is_empty():
      return None
    value = self.front.value
    self.front = self.front.next
    if self.front is None:
      self.end = None
    self.size -= 1
    return value

  def peek(self):
    if self.is_empty():
      return None
    return self.front.value

  def is_empty(self):
    return self.front is None  # Returns true if front is None

  def __len__(self):
    return self.size

  def __str__(self):
    values = []
    current = self.front
    while current:
      values.append(str(current.value))
      current = current.next
    return "\n".join(values)
```

---

## Usage examples

Some common uses of queues:

- Print queues where first in, first printed.
- Task queues in operating systems for execution order.
- Simulations where arrival order must be respected like in banks.
- Message queues like RabbitMQ or Kafka.
- Circular buffers in audio for streaming.

---

## Conclusion

Queues are versatile structures thanks to their FIFO principle. Having a good handle on queues, implementation, and applications will reinforce your skills as a programmer.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}
