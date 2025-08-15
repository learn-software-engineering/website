---
draft: true
weight: 4
series: ["Learning Software Engineering"]
series_order: 4
title: "SDLC: The System Design phase"
authors:
  - jnonino
description: >
  System design is an essential part of the Software Development Life Cycle (SDLC). In many ways, this phase sets the stage for the eventual construction of the software system.
date: 2023-08-13
tags: ["Software Engineering", "Software Development Life Cycle", "SDLC", "System Design"]
---

It's the stage where we transition from understanding what the software should do (requirements) to how it will achieve this[^1]. It provides the architectural blueprints for the software, establishing the foundation upon which code is written, databases are structured, and systems communicate. It's a vital roadmap for developers, ensuring everyone understands and aligns on how the system will operate.

System design typically breaks down into two sub-categories:

- **High-Level design (HLD)**: think of this as the macro view. It focuses on system architecture, representing the main components, their relationships, and interactions[^2]. It’s where we decide on the software's major structural elements.

- **Low-Level design (LLD)**: now, zooming into the micro view. LLD delves into each module's intricacies, defining attributes, methods, and detailing how they'll function together[^3]. This stage translates broader system architecture into actionable tasks and modules.

---

## Components

The holistic approach to system design is built upon several pillars:

### Data design

Data is often the lifeblood of systems. Whether it's user profiles on a social media platform or transaction records in a banking system, data shapes software function. In data design:

- We determine the **data structures**, deciding whether to use arrays, linked lists, trees, or graphs[^4].
- We outline **data storage**, choosing between relational databases, NoSQL stores, or hybrid solutions.
- **Data flow** is mapped out, ensuring smooth data movement through the software components.

### Architecture design

This is where the system's high-level structure comes to life. Several decisions and considerations form this design:

- Choosing the right **software architectural pattern**, such as client-server, three-tier, or microservices[^5].
- Factoring in scalability, flexibility, and responsiveness. For instance, should the system be designed to handle spikes in user requests?
- Deciding on the interaction between software components, which can be synchronous or asynchronous.

### Interface design

In today's interconnected digital world, few systems operate in isolation. Interface design ensures our software communicates effectively with other entities:

- It defines **API endpoints**, standardizing how external systems request data or operations[^6].
- Details the format of data interchange, whether it's JSON, XML, or other formats.
- Ensures error handling in interactions, gracefully managing failed data requests or system operations.

### Procedural design

While the architecture offers a structural view, procedural design dives into operational specifics:

- It maps out the **system's logic**, defining algorithms and methodologies[^7].
- We detail the sequence of operations, ensuring the system processes flow effectively.
- Considers potential system bottlenecks and ensures efficiency in execution.

---

## Tools and techniques

Several tools aid software engineers in crafting and visualizing system design:

**Unified Modelling Language (UML)**

A universally recognized notation, UML offers varied diagrams from showcasing system interactions (use-case diagrams) to detailing structure (class diagrams)[^8].

**Entity-Relationship Diagrams (ERD)**

Primarily for data design, ERDs visually represent database logical structures[^9].

**Flowcharts**

For procedural design, flowcharts offer a graphical representation of processes, making complex algorithms more digestible.

---

## Why it is important

A robust system design is pivotal for several reasons:

- It provides clarity, ensuring developers understand the roadmap ahead[^10].
- Helps in early detection of system flaws or bottlenecks.
- Establishes a foundation that can adapt to evolving requirements or technological shifts.

---

## Conclusion

In conclusion, system design stands as the bridge between conceptual requirements and tangible software. As aspirant software engineers, investing time to grasp and master this phase can be the difference between crafting software that merely works and software that excels.

---

## References

[^1]: Sommerville, I., 2010. Software Engineering. 9th ed. Pearson.
[^2]: Booch, G., 2007. Object-Oriented Analysis and Design with Applications. 3rd ed. Addison-Wesley.
[^3]: Pressman, R.S., 2010. Software Engineering: A Practitioner's Approach. 7th ed. McGraw-Hill.
[^4]: Tanenbaum, A.S., 2012. Structured Computer Organization. 6th ed. Pearson.
[^5]: Bass, L., Clements, P., & Kazman, R., 2012. Software Architecture in Practice. 3rd ed. Pearson.
[^6]: Fielding, R.T., 2000. Architectural Styles and the Design of Network-based Software Architectures. Doctoral dissertation, University of California, Irvine.
[^7]: Page-Jones, M., 1988. The Practical Guide to Structured Systems Design. 2nd ed. Yourdon Press.
[^8]: Fowler, M., 2004. UML Distilled: A Brief Guide to the Standard Object Modelling Language. 3rd ed. Addison-Wesley.
[^9]: Chen, P.P., 1976. The Entity-Relationship Model: Toward a Unified View of Data. ACM Transactions on Database Systems, 1(1).
[^10]: Royce, W.W., 1987. Managing the Development of Large Software Systems. Proceedings of IEEE WESCON.
