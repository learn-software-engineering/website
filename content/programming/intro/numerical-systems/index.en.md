---
weight: 3
series: ["Introduction to Programming"]
series_order: 3
title: "Numerical Systems"
authors:
  - jnonino
description: >
  Every day, we're surrounded by numbers. From the alarm clock's digits waking us up in the morning to the price of our favourite morning coffee. But, have you ever stopped to ponder the essence of these numbers? In this article, we will dive deep into the captivating world of numbering systems, unravelling how one number can have myriad representations depending on the context.
date: 2023-09-05
tags: ["Programming", "Numerical Systems", "Decimal", "Binary", "Hexadecimal"]
---

## The decimal system: the bedrock of our daily life

From a tender age, we're taught to count using ten digits: 0 through 9. This system, known as the decimal system, underpins almost all our mathematical and financial activities, from basic arithmetic to calculating bank interests. Its roots trace back to our anatomy: the ten fingers on our hands, making it the most intuitive and natural system for us. Yet, its true charm emanates from its positional nature.

To grasp this concept, let's dissect the number 237:

- The rightmost digit (7) stands for the *units'* position. That is, \\(7 \times 10^0\\) (any number raised to the power of 0 is 1). Therefore, its value is simply 7.
- The middle digit (3) represents the *tens'* position, translating to \\(3 \times 10^1 = 3 \times 10 = 30\\).
- The leftmost digit (2) is in the *hundreds'* position, decoding to \\(2 \times 10^2 = 2 \times 100 = 200\\).

When these values are combined,

```math
2 \times 10^2 + 3 \times 10^1 + 7 \times 10^0 = 200 + 30 + 7 = 237
```

---

## The binary system: computers' coded language

While the decimal system reigns supreme in our everyday lives, the machines we use daily, from our smartphones to computers, operate in a starkly different realm: the binary world. In this system, only two digits exist: 0 and 1. It might seem restrictive at first glance, but this system is the essence of digital electronics. Digital devices, with their billions of transistors, operate using these two states: on (1) and off (0).

Despite its apparent simplicity, the binary system can express any number or information that the decimal system can. For instance, the decimal number 5 is represented as 101 in binary.

Binary, with its ones and zeros, operates in a manner akin to the decimal system, but instead of powers of 10, it uses powers of 2.

Consider the binary number 1011:

- The rightmost bit denotes \\(1 \times 2^0 = 1\\).
- The subsequent bit stands for \\(1 \times 2^1 = 2\\).
- Next up is \\(0 \times 2^2 = 0\\).
- The leftmost bit in this number signifies \\(1 \times 2^3 = 8\\).

Thus, 1011 in binary translates to the following in the decimal system:

```math
1011 = 1 \times 2^3 + 0 \times 2^2 + 1 \times 2^1 + 1 \times 2^0 = 8 + 0 + 2 + 1 = 11
```

---

## Hexadecimal system: bridging humans and machines

While the binary system is perfect for machines, it can get a tad cumbersome for us, especially when dealing with lengthy binary numbers. Here is where the hexadecimal system, employing sixteen distinct digits: 0-9 and A-F, with A representing 10, B as 11, and so forth, up to F, which stands for 15 comes to help.

Hexadecimal proves invaluable as it offers a more compact way to represent binary numbers. Each hexadecimal digit corresponds precisely to four binary bits. For instance, think of the binary representation of the number 41279 and notice how the hexadecimal system achieves a more succinct representation:

```math
41279 = 1010 0001 0011 1111 = A13F
```

But the hexadecimal system is more than just a compressed representation of binary numbers; it's a positional numbering system like decimal or binary but based on 16 instead of 10 or 2. Let's see how to derive the decimal representation of the example number (A13F):

- The rightmost digit represents \\(F \times 16^0 = 15 \times 16^0 = 15\\).
- The subsequent one stands for \\(3 \times 16^1 = 48\\).
- The next digit denotes \\(1 \times 16^2 = 256\\).
- The leftmost digit in this number signifies \\(A \times 16^3 = 10 \times 16^3 = 40960\\).

Therefore, A13F in hexadecimal translates to the following in the decimal system:

```math
A13F = A \times 16^3 + 1 \times 16^2 + 3 \times 16^1 + F \times 16^0 = 10 \times 4096 + 1 \times 256 + 3 \times 16 + 15 \times 1 = 40960 + 256 + 48 + 15 = 41279
```

---

## Conclusion

Numbering systems are like lenses through which we perceive and understand the world of mathematics and computing. Although the decimal system may be the linchpin of our daily existence, it's crucial to appreciate and comprehend the binary and hexadecimal systems, especially in this digital age.

So, the next time you're in front of your computer or using an app on your smartphone, remember that behind that user-friendly interface, a binary world is in full swing, with the hexadecimal system acting as a translator between that realm and us.

---

## References

1. Ifrah, G. (2000). *The Universal History of Numbers*. London: Harvill Press.
2. Tanenbaum, A. (2012). *Structured Computer Organization*. New Jersey: Prentice Hall.
3. Knuth, D. (2007). *The Art of Computer Programming: Seminumerical Algorithms*. California: Addison-Wesley.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
