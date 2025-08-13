---
title: "Continuous Delivery"
authors:
  - jnonino
description: >
  Here you can find information about Continuous Delivery, what is it, which benefits can you expect from it, the challenges you might face trying to adopted and so on.
date: 2023-09-02
tags: ["Continuous Delivery", "Continuous Deployment", "CD"]
---

Continuous Delivery (CD) is deeply rooted in the first principle of the Agile Manifesto posted in 2001[^1]:

> Our highest priority is to satisfy the customer through early and continuous delivery of valuable software.

According to Martin Fowler[^2], a project is truly embracing Continuous Delivery when:

- The software can be deployed at any stage of its lifecycle.
- The team values keeping the software deployment-ready over adding new features.
- Any software version can be deployed to any environment on-demand.

In many ways, Continuous Delivery is the next step from [Continuous Integration]({{< ref "blog/continuous-integration" >}}), seamlessly automating processes all the way to the software's end user.

Much like in [Continuous Integration]({{< ref "blog/continuous-integration" >}}), the *pipeline* stands as the cornerstone of Continuous Delivery. However, it's not just about integrating code, testing it, and generating a deliverable artifact. With CD, this artifact progresses through a series of test phases in environments that increasingly resemble production settings. After passing through the [Continuous Integration]({{< ref "blog/continuous-integration" >}}) *pipeline*, what emerges is a potentially deployable artifact. CD then takes this artifact and puts it through the necessary tests, ensuring it's primed for a live deployment.

---

## The delivery process

The following figure illustrates the fundamental steps of a Continuous Delivery (CD) process. It's vital to note that an effective CD process presupposes a well-oiled [Continuous Integration (CI)]({{< ref "blog/continuous-integration" >}}) system as its foundation.

{{< figure
    src="cd_process.png"
    alt="CD process flowchart"
    caption="Figure 1: CD process flowchart"
    >}}

The CD process adds two distinct phases to the CI system: **deployment stages** and **system validation and verification testing**. A critical premise is that the deployment process is paramount to product delivery. Thus, automating and frequently testing it is essential. Ideally, testing should also be automated to facilitate execution, especially if you aim for [Continuous Deployment](#continuous-deployment). However, it's not imperative to automate all tests, especially at the beginning of the CD process implementation.

Upon creating a potentially deliverable artifact in the CI *pipeline*, the initial step is to **deploy it in a testing or QA environment**. The primary goal of this environment is to offer a space where software tests, both automatic and manual, can be run. The only possible variations between the QA and actual production environment (the one used by customers) may pertain to its capacity or resource size. This scaled-down capacity, whether in processing, memory, storage, or database size, is typically a cost-saving measure.

If system and acceptance tests are successful in the QA environment, the next phase is to **deploy the artifact in a staging environment**. This environment should be an exact replica of the production setting. Its purpose is to facilitate performance and load testing on the system, effectively putting it under stress to validate its functionality under high loads and assess its processing and response times.

When the previous tests are satisfactory, the artifact is ready to be **deployed in a production environment**. This space is also referred to as the *live* environment, where users interact with the system.

As evident, the product deployment method is exercised at least twice before the final production deployment, considerably reducing risks associated with potential errors as the final delivery approaches.

---

## Core principles and practices

One of CD's primary objectives is to boost delivery frequency. This naturally means that each release should **minimize its size**. A smaller delivery not only implies fewer potential errors but also facilitates their identification and resolution[^2]. To achieve this, the code must always be in a *deploy-ready* state.

**Rapid automated testing** is crucial for today's software development[^3]. In the context of CD, automated tests are of utmost importance as they must ensure comprehensive system quality without hampering the delivery cycle's speed[^4]. Lengthy test runs force the team to await results, leading to wasted time[^3]. If these tests aren't parallelized and optimized, developers may start sidelining them[^5]. To ensure test optimization, it's recommended to fail tests that exceed a reasonable limit[^6].

The **software should always be potentially deliverable**. It's essential for the software's build status to remain *'green'* consistently. This means if a new repository commit breaks the build, this error must be rectified before any new commit is made. Adhering to this practice simplifies error tracing, while deviating may condition developers to overlook a broken or *'red'* build[^6]. Broken builds signify product flaws and render it undeliverable.

In using CD, the aim is to ensure each commit results in a *production-ready* artifact. If this artifact is rebuilt at every process stage, there's no guarantee of their identical nature, and validation and verification test results from one artifact cannot be applied to others. This could lead to deploying untested artifacts[^6]. Hence, **artifacts should be produced only once** and undergo all tests on that single artifact before deploying that very software piece to production.

Software should **always be deployed identically across all environments**. Otherwise, there's no assurance of the deployment process's efficacy. If environment differences exist, they should be managed using configuration files, but the deployment process should remain consistent[^6].

**Deployment should be achievable with a mere button press**. The ability to deploy the latest product version anytime with a single button press is a robust indicator of CD implementation[^2]. This not only requires the aforementioned steps but also mandates version-controlled software deployment scripts that are regularly checked and validated[^6].

As mentioned earlier, CD's main objective is to ascertain the product's readiness post each commit[^2]. For this, aside from ensuring a swift compilation and testing process, **process outcomes should be visible and shared with the team**. Display screens showcasing the current status, build times, test coverage, and more, offer a way to maintain constant visibility[^6]. Presenting these results not only supports CD but, as some authors suggest, also bolsters team motivation[^4].

---

## Benefits of Continuous Delivery

With the rise of Continuous Delivery (CD) practices, businesses are realizing numerous advantages:

1. **Swift value delivery**: the adoption of CD practices facilitates a faster rollout of features and error rectifications to end users[^7]. This speed is attributed to the code becoming potentially deployable to users as it traverses through the [Continuous Integration (CI)]({{< ref "blog/continuous-integration" >}}) pipeline[^4].

2. **Diminished delivery failures**: frequent releases lead to **fewer delivery risks** since each release embodies fewer changes. This, in turn, means there's a reduced margin for errors[^2]. Even if an error emerges, its origin is easier to trace and correct. Plus, reverting to a previous version in the face of complications becomes straightforward[^8]. A delightful by-product of this streamlined process is the diminished stress levels amongst stakeholders[^7] [^5]. Furthermore, the consistency and quality in the delivery process cement trust between the development team and clients[^7].

3. **Productivity amplification**: Chen's study reveals that, in the absence of CD practices, teams spent nearly \\(20%\\) of their time setting up and maintaining environments[^7]. The introduction of CD automates this ordeal, which translates to **heightened productivity** by phasing out manual, non-value-add tasks[^6].

4. **Prompt feedback**: a consensus among experts highlights the value of **early feedback** during the development phase[^2] [^7] [^4]. Regular releases not only fetch this feedback promptly but also ensure that the developed product aligns seamlessly with customer requirements[^2]. The agility to swiftly address customer needs, fix reported bugs, and introduce demanded features can significantly elevate customer satisfaction[^4].

5. **Enhanced software quality**: the rigorous automation of software tests (unit, integration, and system), essential for implementing CD, combined with more frequent software deliveries encapsulating fewer changes, culminates in a **notable boost in software quality**[^4].

6. **Transparency in progress**: thanks to frequent releases, clients find it easier to stay abreast of the latest product updates, embracing novel features and bug fixes with open arms.

---

## Challenges in adopting Continuous Delivery

The journey to incorporate Continuous Delivery (CD) isn’t without its hurdles. Here's a look at some of the key challenges faced by businesses:

- **Product complexity**: certain products consist of myriad interconnected modules or possess dependencies with other projects. This can create bottlenecks when automating CD pipelines[^4]. Such complexities might frustrate teams, pushing them to retain manual processes or longer integration procedures.

- **Test suite creation**: crafting a **comprehensive test suite** to ensure product quality is labour-intensive. All team members must allocate significant time to pen tests, which might require extra training. Besides, tests can be time-consuming, so the trick lies in devising effective yet swift tests[^4].

- **Legacy code**: systems that have been in development for an extended period and weren't designed with automated testing in mind can pose a significant challenge. Transitioning such systems to CD is not only a technical endeavour but also a social and cultural shift[^4].

- **Environmental discrepancies**: all environments in the CI pipeline should mirror the production setting (*'production-like'*)[^6]. If not, unforeseen errors can arise, demanding valuable time on non-value-add tasks[^4]. Ensuring multiple deployment stages in similar environments means that the deployment process is tested repetitively.

- **Client restrictions**: not every client might desire or require a shortened product delivery cycle[^4]. Such client limitations aren't direct obstacles to CD implementation but play a pivotal role when aiming for [**Continuous Deployment**](#continuous-deployment). This could potentially sideline some CD benefits, like swift client feedback or minimal change deliveries. Moreover, the domain itself can act as a bottleneck. Software related to heavily-regulated sectors like healthcare and defence may find CD implementation challenging, if not nearly impossible[^4].

- **Transparency and reporting**: a successful CD adoption hinges on collaboration and transparency, introducing challenges tied to **providing effective status reports** to stakeholders[^9]. This encompasses technical hurdles around automated report generation and analytics challenges to pinpoint essential information.

- **Resistance to change**: any transformative change, like adopting CD, faces the innate human trait of **resistance**. Convincing the team to adapt to new practices and modify their workflow is a task. On a higher level, management might also be hesitant to experiment with fresh processes.

---

## Continuous Deployment

The notion of **Continuous Deployment** was popularized by Fitz[^10]. The standout distinction between this method and Continuous Delivery (CD) is that, once the product is deemed ready for release, there's no waiting for manual intervention to launch it into production[^11] [^6]. If CD is properly executed, the step towards Continuous Deployment should demand no extra efforts from the development perspective[^8]. In essence, the core difference lies in decision-making: with CD, releasing the product to production remains a business decision, whereas with Continuous Deployment, the launch is automated.

It's worth noting that when we discuss CD, it inherently includes **Continuous Delivery** but may or may not encompass **Continuous Deployment**.

---

## References

[^1]: Beck, Kent, Mike Beedle, Arie van Bennekum, Alistair Cockburn, Ward Cunningham, Martin Fowler, James Grenning, Jim Highsmith, Andrew Hunt, Ron Jeffries, Jon Kern, Brian Marick, Robert C. Martin, Steve Mellor, Ken Schwaber, Jeff Sutherland, and Dave Thomas. [Manifesto for Agile Software Development](http://agilemanifesto.org/), 2001.

[^2]: Fowler, Martin. [Continuous Delivery](https://martinfowler.com/bliki/ContinuousDelivery.html?utm_source=Codeship&utm_medium=CI-Guide), 2013.

[^3]: Udd, Raoul. Adopting Continuous Delivery: A Case Study, 2016.

[^4]: Leppanen, Marko, Simo Makinen, Max Pagels, Veli-Pekka Eloranta, Juha Itkonen, Mika V. Mantyla, and Tomi Mannisto. [The highways and country roads to continuous deployment](http://ieeexplore.ieee.org/document/7057604/). IEEE Software, 32(2):64-72, 2015.

[^5]: Neely, Steve, and Steve Stolt. [Continuous delivery? Easy! Just change everything (well, maybe it is not that easy)](https://doi.org/10.1109/AGILE.2013.17). Proceedings - AGILE 2013, pp.121-128, 2013.

[^6]: Humble, Jez, and David Farley. [Continuous Delivery: Reliable Software Releases through Build, Test and Deployment Automation](https://my.safaribooksonline.com/9780321601919). Addison-Wesley, 2011.

[^7]: Chen, Lianping. [Continuous Delivery: Huge Benefits, but Challenges Too](http://ieeexplore.ieee.org/document/7006384/). IEEE Software, 32(2):50-54, 2015.

[^8]: Humble, Jez. [Continuous Delivery vs Continuous Deployment](https://continuousdelivery.com/2010/08/continuous-delivery-vs-continuous-deployment/), 2010.

[^9]: Olsson, Helena Holmstrom, Hiva Alahyari, and Jan Bosch. [Climbing the "Stairway to Heaven" -- A Multiple-Case Study Exploring Barriers in the Transition from Agile Development towards Continuous Deployment of Software](http://ieeexplore.ieee.org/document/6328180/). 2012 38th Euromicro Conference on Software Engineering and Advanced Applications, pp.392-399, 2012.

[^10]: Fitz, Timothy. [Continuous Deployment](http://timothyfitz.com/2009/02/08/continuous-deployment/), 2009.

[^11]: Pulkkinen, Ville. Continuous Deployment of Software. Proceedings of Cloud-Based Software Engineering, pp.46-52, 2013.


ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBbKWwWQESzQtuMvqa/V0GFc4I5fo/4xYPinLdLeCot/
