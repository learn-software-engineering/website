---
weight: 4
series: ["Data Structures"]
series_order: 4
title: "Linked Lists"
authors:
  - jnonino
description: >
  Linked lists are a dynamic and flexible data structure that allows efficiently storing collections of elements. Unlike arrays, linked lists do not require reserving a fixed size in advance.
date: 2023-11-01
tags: ["Programming", "Data Structures", "Lists", "Linked Lists"]
---

A linked list is composed of `nodes` where each node has two parts:

- Data or information
- Reference to the next node

Nodes are organized sequentially, each pointing to the next. The last node points to null to indicate the end.

This dynamic structure allows efficient insertion and deletion of nodes.

---

## Types of linked lists

There are several types:

- **Singly linked:** Each node points to the next one. Useful for queues and stacks.

{{< figure
    src="images/linked-list-single.jpg"
    alt="Diagram of a single linked list"
    caption="Diagram of a single linked list"
    >}}

- **Doubly linked:** Each node has reference to the next and previous ones. Allows traversing in both directions.

{{< figure
    src="images/linked-list-double.jpg"
    alt="Diagram of a doubly linked list"
    caption="Diagram of a doubly linked list"
    >}}

- **Circular:** The last node points to the first forming a cycle. Useful for circular buffers.

{{< figure
    src="images/linked-list-circular.jpg"
    alt="Diagram of a circular linked list"
    caption="Diagram of a circular linked list"
    >}}

---

## Common operations

- **Insert:** Add nodes to the beginning, end or middle of the list.

- **Delete:** Remove nodes by value or position.

- **Search:** Find nodes by value by traversing the list.

- **Traverse:** Iterate through nodes by following the references.

---

## Implementation

Linked lists can be implemented as follows:

Use the `ListNode` class to represent nodes:

```python
class ListNode:
  def __init__(self, value):
    self.value = value
    self.next = None
```

Then to create and use a list define a LinkedList class with methods for the operations:

```python
class LinkedList:
  def __init__(self):
    self.head = None

  def add_to_start(self, new_value):
    new_node = ListNode(new_value)
    new_node.next = self.head
    self.head = new_node

  def print(self):
    current = self.head
    while current != None:
      print(current.value)
      current = current.next

  def search(self, searched_value):
    current = self.head
    while current != None:
      if current.value == searched_value:
        return True
      current = current.next
    return False

  #...other methods
```

With this `LinkedList` class we can create a list, add nodes, print it, search elements, etc.

We could add other methods like insert at end, delete by value, get by index, etc. But this gives an idea of how to encapsulate the linked list functionality in a class. As practice, feel free to try adding these methods on your own. Don't hesitate to leave your comments and contact me if you need help!

---

## Advantages and disadvantages

**Advantages:**

- Inserting and deleting nodes is efficient.
- Doesn't require defining a fixed size like arrays.
- Dynamic, flexible structure.

**Disadvantages:**

- More memory usage from having to store references.
- Accessing elements by index is slower since it is sequential.

---

## Usage examples

- Implementing structures like stacks and queues.
- In doubly linked lists, traverse the list in both directions.
- Blockchains like Bitcoin's.
- Playing elements in order like music playlists.

---

## Conclusion

Linked lists are a versatile data structure for storing dynamic sequences of elements. Having a good handle on these lists, their operations, and uses is essential for any programmer.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}
