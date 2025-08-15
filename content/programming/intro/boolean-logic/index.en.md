---
weight: 4
series: ["Introduction to Programming"]
series_order: 4
title: "Boolean Logic"
authors:
  - jnonino
description: >
  In life, we often seek certainties. Will it rain tomorrow, true or false? Is a certain action right or wrong? This dichotomy, this division between two opposing states, lies at the very core of a fundamental branch of mathematics and computer science: Boolean logic.
date: 2023-09-09
tags: ["Programming", "Boolean Logic"]
---

Named in honour of [George Boole](https://en.wikipedia.org/wiki/George_Boole), a 19th-century English mathematician, Boolean logic is a mathematical system that deals with operations resulting in one of two possible outcomes: true or false, typically represented as 1 and 0, respectively. In his groundbreaking work, "An Investigation of the Laws of Thought," Boole laid the foundations for this logic, introducing an algebraic system that could be employed to depict logical structures.

---

## Boolean operations

Within Boolean logic, several fundamental operations allow for the manipulation and combination of these binary expressions:

1. **AND**: This operation yields true (1) only if both inputs are true. For instance, if you have two switches, both need to be in the *on* position for a light to illuminate.

2. **OR**: It returns true if at least one of the inputs is true. Using the switch analogy, as long as one of them is in the *on* position, the light will shine.

3. **NOT**: This unary operation (accepting only one input) simply inverts the input value. Provide it with a 1, and you'll get a 0, and vice versa.

4. **NAND (NOT AND)**: It's the negation of AND. It only returns false if both inputs are true.

5. **NOR (NOT OR)**: The negation of OR. It yields true only if both inputs are false.

6. **XOR (Exclusive OR)**: It returns true if the inputs differ. If both are the same, it returns false.

7. **XNOR (Exclusive NOR)**: The inverse of XOR. It yields true if both inputs are the same.

## Why is this logic important in computing and programming?

Modern computing, at its core, is all about bit manipulation (those 1s and 0s we've mentioned). Every operation a computer undertakes, from basic arithmetic to rendering intricate graphics, involves Boolean operations at some level.

In programming, Boolean logic is used in control structures, such as conditional statements (if, else) and loops, allowing programs to make decisions based on specific conditions.

---

## Truth Tables: mapping Boolean logic

A truth table graphically represents a Boolean operation. It lists every possible input combination and displays the operation's result for each combination.

For instance:

| A | B | A AND B | A OR B | A XOR B | A NOR B | A NAND B | NOT A | A NXOR B |
|:-:|:-:|:-------:|:------:|:-------:|:-------:|:--------:|:-----:|:--------:|
| 1 | 1 | 1       | 1      | 0       | 0       | 0        | 0     | 1        |
| 1 | 0 | 0       | 1      | 1       | 0       | 1        | 0     | 0        |
| 0 | 1 | 0       | 1      | 1       | 0       | 1        | 1     | 0        |
| 0 | 0 | 0       | 0      | 0       | 1       | 1        | 1     | 1        |

---

## Concluding thoughts

Boolean logic is more than a set of abstract mathematical rules. It's the foundational language of machines, the code underpinning the digital age in which we live. By understanding its principles, not only do we become more proficient in working with technology, but we also gain a deeper appreciation of the structures supporting our digital world.

---

## References

1. Boole, G. (1854). *An Investigation of the Laws of Thought*. London: Walton and Maberly.
2. Tanenbaum, A. (2012). *Structured Computer Organization*. New Jersey: Prentice Hall.
3. Minsky, M. (1967). *Computation: Finite and Infinite Machines*. New Jersey: Prentice-Hall.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
