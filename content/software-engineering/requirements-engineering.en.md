---
draft: true
weight: 9
series: ["Software Engineering"]
series_order: 9
title: "Requirements Engineering"
authors:
  - jnonino
description: >
  Requirements engineering is the process of defining, documenting and maintaining requirements for a software system. It is the first and foundational stage in the software development lifecycle. Requirements express the needs and constraints that the software must satisfy in order to solve problems and provide value to users and the business. Let's get started understanding the basics of requirements in software projects.
date: 2023-08-18
tags: ["Software Engineering", "Requirements Engineering", "Requirements", "SRS"]
---

According to Sommerville and Sawyer, a requirement is a statement about an intended product that specifies what it should do or how it should perform[^1].

Requirements identify the goals, needs and constraints that the software must address in order to deliver value. They are a form of specification that serve as the foundation for all subsequent software engineering activities.

Some key definitions of requirements are:

- Description of how the system should behave.
- Features that the software needs to deliver.
- Functionalities that users expect from the software.
- Statements of business rules that software must comply with.
- Constraints on development of the software.

Requirements convey the intended purpose, context and environment in which the software will operate. They express the underlying motivations for why the software needs to be built by identifying problems faced by users or organization.

As Wiegers puts it, *requirements describe the inner and outer capabilities, characteristics, and qualities that the product must have or perform well to satisfy all aspects of the product vision*[^2].

## What should requirements define?

Good requirements provide a complete description of **what** the software will do, without dealing with **how** it will be implemented.

Some key aspects that requirements should define are:
- **User capabilities**: what can users do with the system? e.g. login, search, make payment etc.
- **Operational workflows**: how does system interact with users and external systems? e.g. order processing workflow.
- **Business rules**: what policies and rules does system need to comply with? e.g. credit limits.
- **Integration touch points**: how does system integrate with other systems? e.g. payment gateway APIs.
- **Quality attributes**: what non-functional qualities are important? e.g. security, availability.
- **External interfaces**: how does system interact with external actors? e.g. device interfaces.
- **Environment conditions**: what environmental constraints are present? e.g. operating system.
- **Design/architectural constraints**: are there constraints on design choices? e.g. use cloud infrastructure.

Documenting these aspects ensures requirements provide a good view into the problem space and what the software solution needs to achieve.

## Properties of good requirements

High quality requirements exhibit certain desired characteristics:

- **Complete**: requirements capture the full capabilities needed without any gaps.
- **Correct**: requirements precisely represent genuine user needs.
- **Feasible**: requirements can be reasonably implemented within constraints.
- **Unambiguous**: requirements have clear meaning that is open to only one interpretation.
- **Consistent**: there are no conflicts or contradictions between requirements.
- **Prioritized**: importance and urgency is indicated to guide implementation.
- **Verifiable**: requirements can be validated via concrete acceptance criteria.
- **Traceable**: origin of each requirement is clear for analysis.
- **Atomic**: requirements define one need instead of multiple combined needs.
- **Implementation free**: requirements avoid implementation choices or design details.

Adhering to these qualities results in a strong set of requirements that serve as a solid foundation for the software design and development.

---

## Levels of requirements

Requirements are defined at different levels of abstraction and detail. Requirements evolve from high-level needs to detailed specifications through this hierarchical breakdown.

### Business requirements

These capture high-level needs related to business objectives, problems and outcomes the software should achieve. They are written using terminology familiar to business users. Examples include:

- Improve customer retention by 10%.
- Reduce operational costs by optimizing order processing.
- Enable omni-channel order fulfilment.

### User requirements

These describe the needs and wishes of end-users related to functionality and quality. They capture who, what, where, when and why of user interactions. Examples:

- User shall be able to place orders 24/7 via website.
- System shall provide real-time order status tracking.
- User shall be able to save payment information securely.

### System requirements

These describe the detailed behaviours, operations, capabilities and business rules the system needs to implement. Or, impose quality constraints, performance needs and other attributes on the system. Examples:

- System shall verify customer identity before accepting order.
- User shall be able to filter orders by date, status, amount, etc.
- System shall interface with the inventory system to validate product availability.
- System must have a uptime of 99.95%.
- Page response time must be under 100 ms for 90% of requests.
- System must store customer data encrypted at rest.

---

## Types of requirements

There are two main types of requirements:

### Functional requirements

These describe the specific behaviours and operations that the system needs to perform in order to accomplish the intended purpose. Functional requirements depend heavily on the type of software being developed. Some examples of functional requirements are:

- The system shall allow users to sign in using their email address and password.
- The system shall allow admins to add new user accounts.
- The system shall allow managers to generate sales reports by region.
- The system shall integrate with PayPal to accept payments.

### Non-functional requirements

Non-functional requirements specify quality attributes and constraints that software must meet. They impose restrictions on **how** the system performs its functions. Examples of non-functional requirements:

- The system shall have an uptime of 99.95%.
- The system shall support up to 100 concurrent active users.
- User passwords must be stored in encrypted format.
- The website shall work on mobile and desktop browsers.

In turn, non-functional requirements can be divided into the following categories. Covering relevant categories of non-functional needs results in a comprehensive set of quality requirements.

#### Product Requirements

These relate to quality characteristics of the software itself. Types include:

- **Usability**: how easy is it for users to learn and operate the system?
- **Performance**: response times, throughput, resource usage, scalability targets, etc.
- **Security**: access control, encryption, vulnerability prevention, etc.
- **Availability**: uptime, reliability, disaster recovery needs, etc.
- **Maintainability**: ability to make changes, add new features, repair defects, etc.
- **Portability**: ability to work across different environments and platforms.

#### Organizational Requirements

These relate to business, process and compliance needs of the organization. Types include:

- **Delivery**: schedule, release planning, milestones to be met.
- **Legal**: regulations, policies, standards that must be complied with.
- **Data**: data formats, retention periods, reporting needs, etc.
- **Process**: integration with organizational workflows, change management, etc.
- **Operational**: server environments, infrastructure standards, monitoring needs, etc.

#### External Requirements

These relate to integration with external systems and end user environments. Types include:

- **Interfaces**: protocols, APIs, formats for exchanging data.
- **Interoperability**: ability to integrate with other systems as needed.
- **Hardware**: support for client hardware like scanners, handhelds, etc.
- **Platform**: compatibility with end user operating systems.

---

## Software Requirements Specification (SRS)

The Software Requirements Specification (SRS) is the official document that captures the complete set of requirements for the system in detail. It is treated as the contract between customers and developers.

According to IEEE, the SRS can be defined as *a document that describes all data, functional and behavioural requirements; analyses, specifies, and validates the system requirements*[^3].

Some key purposes that the SRS serves are:

- Formally documents the requirements agreed upon by stakeholders.
- Serves as reference for planning, estimating and tracking the project.
- Reduces ambiguity by recording details about requirements.
- Serves as basis for design, development and testing activities.
- Facilitates communication and consensus between teams.
- Becomes basis for verification and validation.
- Evolves into system tests, user manuals, etc.

The main users of the SRS document are:

- **Customers**: clarify needs, review and approve requirements.
- **Managers**: estimate effort, plan activities, track progress.
- **Architects**: analyze requirements to design solutions.
- **Developers**: understand detailed needs to implement.
- **Testers**: derive test conditions and test cases.
- **Technical Writers**: use it as foundation for user manuals.

A typical SRS is organized into the following sections:

1. **Introduction**: purpose, scope, definitions, acronyms, references.
2. **User Requirements**: capabilities, user characteristics, constraints.
3. **Functional Requirements**: functionalities, operations, business rules.
4. **Non-functional Requirements**: qualities, performance, attributes.
5. **External Interface Requirements**: interactions with other systems, devices, etc.
6. **Other Requirements**: licensing, compliance, platform, deliverables.
7. **Appendix**: diagrams, analysis models, schema definitions.

The SRS maintains traceability between stakeholder needs, requirements specification and system design. This enables tracking of requirements throughout the product lifecycle.

---

## Notations for documenting requirements

Textual description is the most common way of documenting requirements in the SRS. However, visual models and formal notations can also be used to express specific aspects precisely. Here there are different ways of expressing software requirements.

- **User stories**: used in agile methods to capture user requirements. Written as *"As a <user role>, I want <capability> so that <value>"*.
- **Use cases**: describe interactions between users and the system to achieve a goal.
- **Data models**: express structure of data as entities, attributes, relationships etc. e.g. ER diagrams.
- **Behaviour models**: represent dynamic behaviour and workflows e.g. state machine diagrams, activity diagrams.
- **Interface models**: specify external interfaces using API blueprints, protocol specs, etc.
- **Supplementary specifications**: detailed non-functional specs e.g. security standards, compliance rules.

Choosing appropriate techniques allows capturing complete details in an unambiguous manner. The textual SRS can reference these as needed.

---

## Eliciting and analysing requirements

Requirements come from analysing the needs of various stakeholders. Typical steps are:

1. **Stakeholder identification**: identify various stakeholders - end users, business managers, operations team, etc.
2. **Requirements gathering**: gather needs via interviews, observation, focus groups, workshops, surveys, etc. Requirements gathering relies heavily on techniques like prototyping, scenario analysis, domain modelling and qualitative user research.
3. **Requirements analysis**: analyze requirements for clarity, conflicts and prioritization.
4. **Requirements documentation**: document requirements, review with stakeholders and baseline.
5. **Requirements validation**: validate requirements to ensure consistency, completeness, technical feasibility, etc.
6. **Requirements management**: manage changes through proper change control processes. According to Davis, requirements management is critical for aligning software capabilities with business objectives throughout the product lifecycle[^4].

---

## Challenges

Some key challenges faced in requirements engineering are:

- Incomplete, unclear and ambiguous requirements.
- Infeasible requirements beyond project constraints.
- Conflicting stakeholder needs and priorities.
- Frequent requirement changes leading to scope creep.
- Missing key details and assumptions.
- Lack of customer availability and engagement.
- Users not knowing what they need.
- Weak change management discipline.
- Communication gaps across customer and development teams.

According to Christel and Kang, requirements elicitation is a complex activity due to factors like vague customer needs, lack of user involvement and staggered feedback[^5]. Using proven elicitation techniques, sound documentation practices, rigorous analysis and reviews, and disciplined change management helps tackle these challenges.

---

## Emerging Trends

Some key trends shaping the future of requirements engineering:

- Increased end user involvement through prototyping and crowdsourcing.
- Growing use of iterative approaches to allow for emergent requirements.
- Greater reliance on product managers as customer proxies.
- Increased adoption of agile user stories over traditional SRS docs.
- Enhanced traceability through modelling and automation.
- Integration with design models and testing to realize "single source of truth".
- Introducing requirements management tools for collaboration.
- Leveraging AI techniques for analysing and deriving complex requirements.
- Focus on continuously capturing and refining requirements throughout product lifecycle.

According to Nuseibeh and Easterbrook, requirements engineering will need to adapt to factors like faster delivery cycles, evolving design approaches and increased automation[^6].

---

## Conclusion

Defining complete, correct requirements that accurately capture end user and business needs is essential to delivering effective software solutions. Requirements engineering done right lays the foundation for developing a product that delivers maximum value. Applying sound specification techniques, following collaborative elicitation processes, documenting requirements clearly, and managing changing needs systematically are key disciplines that lead to high quality software products that meet customer expectations.

---

## References

[^1]: Sommerville, Ian, and Pete Sawyer. Requirements engineering: a good practice guide. John Wiley & Sons, Inc., 1997.
[^2]: Karl E. Wiegers, and Joy Beatty. Software requirements. Pearson Education, 2013.
[^3]: IEEE. IEEE Guide for Developing System Requirements Specifications. IEEE Std 1233, 1998 edition.
[^4]: A. M. Davis. Just enough requirements management: where software development meets marketing. Dorset House Publishing Co., Inc., 2005.
[^5]: Christel, Michael G., and Kyo C. Kang. Issues in requirements elicitation. Carnegie Mellon University, 1992.
[^6]: Nuseibeh, Bashar, and Steve Easterbrook. "Requirements engineering: a roadmap." Proceedings of the Conference on the Future of Software Engineering. 2000.
[^7]: Young, Ralph R. Effective requirements practices. Addison-Wesley Professional, 2001.
[^8]: Hull, Elizabeth, et al. Requirements engineering. Springer London, 2011.
[^9]: Laplante, Phillip A. Requirements engineering for software and systems. CRC Press, 2017.
[^10]: Wiegers, Karl, and Joy Beatty. Software Requirements, Third Edition. Microsoft Press, 2013.
