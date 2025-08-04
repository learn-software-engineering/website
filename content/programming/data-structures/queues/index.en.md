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

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/main/programming/data_structures/queues.py"
    type="python"
    >}}

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
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}
