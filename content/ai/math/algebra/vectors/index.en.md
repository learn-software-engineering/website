---
weight: 1
title: "Vectors, Scalars and Vector Spaces"
authors:
  - jnonino
description: >
  Master vectors, scalars & vector spaces from scratch. Step-by-step math derivations, runnable Python code, and research-grade insights for aspiring Machine Learning (ML) scientists.
date: 2026-03-06
tags: ["AI", "Maths", "Algebra", "Vectors"]
---

In 2017, researchers at Google Brain published [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762), introducing the Transformer architecture that now underlies GPT-4, Gemini, and virtually every state-of-the-art language model. At the heart of that architecture (and of every neural network, recommendation system, and computer vision model) is a deceptively simple object: the **vector**.

When a language model reads the word *"bank"*, it doesn't see a string. It sees a vector in a 4096-dimensional space where *"bank (financial)"* and *"bank (riverbank)"* occupy measurably different regions. When a search engine decides that your query matches a document, it is computing an angle between two vectors. When a neural network learns, it is moving vectors through space in response to a gradient, itself a vector.

This article builds your working foundation for all of that. By the end, you will be able to:

- Formally define vectors, scalars, and vector spaces, and explain *why* the axioms matter.
- Compute norms, dot products, and inter-vector angles both by hand and in *NumPy*.
- Reason geometrically about high-dimensional data, a non-negotiable skill for Machine Learning research.
- Read a research paper that uses vector notation without losing the thread.

No fluff, let's start.

## Prerequisites

Before reading this article, you should be comfortable with:

- **High school algebra:** variables, functions, the coordinate plane.
- **Python basics**: lists, loops, functions, importing libraries.
- **Basic calculus intuition** (helpful but not required): the idea that a derivative points in the direction of steepest increase.

## Intuition first

### The programmer's analogy: vectors as typed arrays with geometric soul

As a developer, you've used arrays your whole career. A Python list `[3.0, -1.5, 7.2]` stores three numbers. A vector is superficially the same thing, but with a crucial additional structure: **position in space and the geometry that connects positions**.

Think of it this way. If you have two dictionaries in Python:

```python
user_A = {"age": 28, "purchase_freq": 5, "avg_spend": 120.0}
user_B = {"age": 29, "purchase_freq": 6, "avg_spend": 115.0}
```

As dictionaries, they're just data blobs. You can read values, but *"how similar are these users?"* is not a question the dictionary can answer natively. Now convert them to vectors:

```
A = [28, 5, 120.0]
B = [29, 6, 115.0]
```

Suddenly you have geometry. You can measure the *distance* between them, the *angle* they form relative to the origin, and whether one is a *scaled version* of another. This is the leap vectors make over plain arrays: **they live in a space equipped with rules for measuring, comparing, and transforming**.

### Geometric picture: vectors as arrows

Picture a standard 2D coordinate system. The vector \(\mathbf{v} = [3, 2]\) is an arrow starting at the origin \((0, 0)\) and ending at the point \((3, 2)\). Two things define it completely: its **magnitude** (how long the arrow is) and its **direction** (which way it points).

{{< figure
    src="images/vector.png"
    alt="Visual representation of a two-dimension vector"
    caption="Visual representation of a two-dimension vector"
    >}}

This geometric interpretation is not just visual sugar. In Machine Learning, a data point (a row in your dataset) *is* a vector, it's an arrow in feature space. Two similar data points are arrows pointing in roughly the same direction. An outlier is an arrow pointing somewhere unexpected. Dimensionality reduction (PCA, UMAP) is the art of finding a lower-dimensional space where those arrows still tell roughly the same story.

### Scalars: the simplest case

A **scalar** is just a single number, no direction, no components. Temperature, loss value, learning rate: all scalars. When you multiply a vector by a scalar, you stretch or shrink the arrow without rotating it:

Same direction vector, twice as long.
$$
2 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 3 \\ 2 \cdot 2 \end{pmatrix} = \begin{pmatrix} 6 \\ 4 \end{pmatrix}
$$
```python
2 x [3, 2] = [6, 4]
```

Same direction vector, flipped (180°).
$$
-1 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} -1 \cdot 3 \\ -1 \cdot 2 \end{pmatrix} = \begin{pmatrix} -3 \\ -2 \end{pmatrix}
$$
```python
-1 x [3, 2] = [-3, -2]
```

This operation, **scalar multiplication**, is one of the two foundational operations that define a vector space.

{{< callout type="important" >}}
When debugging a neural network and the loss explodes, it often means vectors (activations or gradients) are being scaled up by factors much greater than 1.0 each layer. Understanding scalar multiplication geometrically helps you see *why* gradient clipping or batch normalization restores stability: they're renormalizing the length of those arrows.
{{< /callout >}}

## Mathematical derivation
