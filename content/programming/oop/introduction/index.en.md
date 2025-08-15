---
weight: 1
series: ["Programming: Learning Object-Oriented Concepts"]
series_order: 1
title: "Introduction"
authors:
  - jnonino
description: >
  Object-Oriented Programming (OOP) is a programming paradigm that has become indispensable nowadays. This approach models real-world elements as "objects" that have properties and behaviours, which allows for more intuitive and maintainable programmes to be created. In this article we will look at the basic concepts of OOP and its advantages over other paradigms like procedural programming.
date: 2025-08-04
tags: ["Programming", "Object-Oriented Programming"]
---

Object-Oriented Programming (OOP) is a programming paradigm that has become indispensable nowadays. This approach models real-world elements as "objects" that have properties and behaviours, which allows for more intuitive and maintainable programmes to be created. In this article we will look at the basic concepts of OOP and its advantages over other paradigms like procedural programming. Let's get started!

This paradigm is based on two fundamental concepts:
- **Objects**: entities that combine state (data) and behaviour (operations) in a single unit. For example, a "car" object would have properties like colour, number of doors, maximum speed, etc. And behaviours like accelerate, brake, steer, etc.
- **Classes**: specifications that define the common structure and behaviour of a group of objects. The "car" class would serve as a template for creating car objects with the same characteristics.

As explained by programmer Alan Kay, one of the creators of OOP:

> "The big idea is to design programmes in terms of conceptual objects and concepts from the real world. The interfaces with the real world should, therefore, be constructed in terms of these conceptual objects."

That is, OOP conceptually models real-world elements to make programming more intuitive.

---

## Programming paradigms

Before delving into OOP, it is worth understanding that there are different paradigms or approaches to tackle programming. The main ones are:

### Procedural programming

Ordered sequence of instructions that the programme must follow step-by-step. The focus is on procedures and functions. For example, C is a language geared towards procedural programming.

Procedural programming is better for:

- Simple problems or sequential algorithms.
- Code that won't need heavy reusing or expanding.
- Cases where performance and efficiency are critical.

### Object-Oriented programming

Model based on objects that contain data and code in cohesive units. The focus is on classes and the interaction between objects. For example, Java and Python are object-oriented languages.

OOP allows modelling real-world elements more directly, better encapsulating data, and reusing code through inheritance between classes.

The main advantages of OOP over procedural programming are:

- **Modularity**: objects group related data and operations, encapsulating internal complexity. This allows working with independent modules.
- **Information hiding**: Objects can expose a simple interface and hide internal implementation details. This reduces coupling.
- **Reusability**: Classes enable code reuse. An abstract class can inherit to multiple subclasses.
- **Extensibility**: We can extend the behaviour of parent classes by creating new subclasses.
- **Conceptual mapping**: Objects represent real-world entities, which eases the translation of requirements into code.

However, OOP also has disadvantages. According to programmer Paul Graham:

>"Object-oriented programming often makes things harder than they need to be."

For example, for simple problems OOP can be excessive. And in large projects there is a risk of overusing inheritance and polymorphism, making the code difficult to follow.

Ultimately, OOP is more suitable when:

- The problem to be modelled has clear, structured entities.
- We want to reuse encapsulated code in modular classes.
- We work on systems that need to be easily extended and maintained.
