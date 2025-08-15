---
draft: true
weight: 8
series: ["Learning Software Engineering"]
series_order: 8
title: "SDLC: The Maintenance phase"
authors:
  - jnonino
description: >
  Once a software application is deployed and begins its life in the hands of users, its journey is far from over. The maintenance stage, often overlooked by novices, plays a pivotal role in the Software Development Life Cycle (SDLC).
date: 2023-08-17
tags: ["Software Engineering", "Software Development Life Cycle", "SDLC", "Maintenance"]
---

This phase ensures that software remains robust, relevant, and valuable throughout its lifecycle[^1].

Software maintenance encompasses all activities post-deployment that are not categorized as software evolution[^2]. It's about refining, enhancing, and optimizing software, as well as fixing errors that surface over time.

While creating software is challenging, maintaining it is often even more demanding. With changing user needs, technology advancements, and the inevitable emergence of bugs, ongoing maintenance is essential for software's continued relevance and efficiency[^1].

---

## Types of software maintenance

- **Corrective maintenance**: This involves identifying and fixing bugs or defects in the software. It's a reactive approach, often prompted by user feedback or system monitoring[^1].
- **Adaptive maintenance**: As external factors like OS updates, hardware changes, or new regulations come into play, software needs to adapt. This type of maintenance ensures the software remains compatible with its changing environment[^3].
- **Perfective maintenance**: Driven by the need to enhance software functionality or performance, this is about adding new features or improving existing ones based on user feedback and evolving market needs[^1].
- **Preventive maintenance**: Proactively looking into the future, this type of maintenance is focused on foreseeing and preventing future issues, often by restructuring or optimizing the code[^4].

---

## The Process of maintenance

1. **Request initiation**: it starts with recognizing the need for maintenance. This can come from user feedback, system alerts, or regular system checks[^1].
2. **Impact analysis**: before making any changes, it's essential to assess the potential impact of the modification on the entire system[^5].
3. **Design and implementation**: with a clear understanding of the task at hand, developers design the solution and implement the necessary changes[^1].
4. **System testing**: post-implementation, rigorous testing is essential to ensure that the changes work as intended and haven’t inadvertently introduced new issues[^2].
5. **Release**: once satisfied, the updated software is released to users, either as patches, updates, or new versions[^1].

---

## Challenges

- **Understanding legacy code**: as software evolves, understanding older parts of the codebase, especially if original developers have moved on, can be challenging[^6].
- **Maintaining documentation**: over time, ensuring that documentation stays up-to-date and accurately reflects the software's current state is a continuous challenge[^7].
- **Balancing new development with maintenance**: allocating resources between developing new features and maintaining existing ones is a tightrope walk many teams grapple with[^8].

---

## The economic aspects

It's often quoted in software engineering literature that maintenance costs can account for over 60% of all SDLC costs[^11]. Hence, budgeting, resource allocation, and cost optimization become integral parts of the maintenance stage.

---

## Conclusion

Maintenance is not merely an afterthought; it's the pulse that keeps software alive and relevant in an ever-changing digital landscape. By understanding and embracing the nuances of this SDLC stage, software engineers can ensure the longevity and success of their creations[^1].

---

## References

[^1]: Sommerville, I., 2010. Software Engineering. 9th ed. Pearson.
[^2]: Pressman, R.S., 2010. Software Engineering: A Practitioner's Approach. 7th ed. McGraw-Hill.
[^3]: Tanenbaum, A.S., 2014. Modern Operating Systems. 4th ed. Prentice Hall.
[^4]: Parnas, D.L., 1994. Software Ageing. *Proceedings of the 16th international conference on Software engineering*. IEEE.
[^5]: Lehman, M.M., 1980. Programs, life cycles, and laws of software evolution. *Proceedings of the IEEE*, 68(9), pp.1060-1076.
[^6]: Bowman, I.T., Holt, R.C., and Brewster, N.V., 1999. Linux as a case study: Its extracted software architecture. *Proceedings of the 21st international conference on Software engineering*. ACM.
[^7]: Lethbridge, T.C., Singer, J., and Forward, A., 2003. How software engineers use documentation: The state of the practice. *IEEE Software*, 20(6), pp.35-39.
[^8]: Boehm, B.W., and Basili, V.R., 2001. Software defect reduction top 10 list. *Computer*, 34(1), pp.135-137.
[^9]: Fowler, M., 2018. Refactoring: Improving the Design of Existing Code. Addison-Wesley Professional.
[^10]: Oram, A., and Wilson, G., 2011. Making Software: What Really Works, and Why We Believe It. O'Reilly Media.
[^11]: Erlikh, L., 2000. Leveraging legacy system dollars for e-business. *IT professional*, 2(3), pp.17-23.
