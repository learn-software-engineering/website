---
weight: 2
series: ["Introduction to Programming"]
series_order: 2
title: "The Computer"
authors:
  - jnonino
description: >
  In today's digital age, where electronic gadgets seamlessly integrate into our daily lives, understanding the bedrock upon which these marvels stand becomes not just an academic interest but a societal necessity. As we embark on this enlightening voyage into the heart of computers, we aim to demystify the intricate dance between the physical and the abstract, between the tangible hardware and the intangible software.
date: 2023-09-02
tags: ["Programming", "Computer", "Hardware", "Software", "CPU", "Memory"]
# images:
#   - /images/banners/content/programming/0100-intro/0101-computers.png
---

To the uninitiated, a computer might seem like a mere box—perhaps sometimes sleek and shiny—but a box nonetheless. Yet, within this "*box*" lies a universe of complexity and coordination.

***Hardware*** represents the physical components of a computer: the Central Processing Unit (CPU) which is often likened to the brain of the system, the Random Access Memory (RAM) acting as a temporary storage while tasks are underway, storage devices that retain data, and peripherals like keyboards, mice, and monitors.

On the other side of this duality is ***software***, a set of instructions that guides the hardware. There are various types of software, from system software like the operating system (OS), which coordinates the myriad hardware components, to application software that allows users to perform specific tasks, such as word processing or gaming.

The role of the operating system is pivotal. It acts as a bridge, translating user commands into instructions that the hardware can execute. If the hardware were an orchestra, the OS would be its conductor, ensuring each instrument plays its part harmoniously.

---

## The binary system: decoding the language of machines

Human civilizations have developed numerous numbering systems over the millennia, but computers, with their logical circuits, have settled on the binary system. But why binary? Simply put, at the most foundational level, a computer's operation is based on switches (transistors) that can be either 'on' or 'off', corresponding naturally to the binary digits, or 'bits', 1 and 0 respectively.

In this binary realm, a *bit* is the smallest data unit, representing a single binary value. A *byte*, comprising 8 bits, can represent 256 distinct values, ranging from \\(00000000\\) to \\(11111111\\). This binary encoding isn't restricted to numbers; it extends to text, images, and virtually all forms of data. For instance, in ASCII encoding, the capital letter 'A' is represented as \\(01000001\\).

In a following post we'll describe in more details the binary system and introduce another system used a lot in relations to computers, the *hexadecimal*.

---

## Memory and Storage: the sanctuaries of data

The concepts of memory and storage are pivotal in understanding computer architecture. Though sometimes used interchangeably in colloquial parlance, their roles in a computer system are distinct.

Memory, particularly RAM, is volatile, meaning information stored is lost once the computer is turned off. RAM serves as the computer's "*workspace*", temporarily storing data and instructions during operations. There are various RAM types, with DRAM and SRAM being the most prevalent.

Contrastingly, Read-Only Memory (ROM) is non-volatile, used predominantly to store firmware—software intrinsically linked to specific hardware, requiring infrequent alterations.

In terms of data storage, devices like hard drives, SSDs, and flash drives offer permanent data retention. These storage mechanisms are part of the memory hierarchy, which ranges from the swift but limited cache memory to the expansive but slower secondary storage.

---

## References

1. Patterson, D. & Hennessy, J. (2014). *Computer Organization and Design*. Elsevier.
2. Silberschatz, A., Galvin, P. B., & Gagne, G. (2009). *Operating System Concepts*. John Wiley & Sons.
3. Tanenbaum, A. (2012). *Structured Computer Organization*. Prentice Hall.
4. Brookshear, J. G. (2011). *Computer Science: An Overview*. Addison-Wesley.
5. Jacob, B., Ng, S. W., & Wang, D. T. (2007). *Memory Systems: Cache, DRAM, Disk*. Morgan Kaufmann.
6. Siewiorek, D. P. & Swarz, R. S. (2017). *Reliable Computer Systems: Design and Evaluation*. A K Peters/CRC Press.

---

{{< alert icon="comment" cardColor="grey" iconColor="black" textColor="black" >}}
Cheers for making it this far! I hope this journey through the programming universe has been as fascinating for you as it was for me to write down.

We're keen to hear your thoughts, so don't be shy – drop your comments, suggestions, and those bright ideas you're bound to have.

Also, to delve deeper than these lines, take a stroll through the practical examples we've cooked up for you. You'll find all the code and projects in our GitHub repository [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Thanks for being part of this learning community. Keep coding and exploring new territories in this captivating world of software!
{{< /alert >}}

---
