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

### Formal definitions

#### Scalar

An element of a field \(\mathbb{F}\), for our purposes, a real number \(\mathbb{R}\) or complex number \(\mathbb{C}\). Denoted with standard italics and usually greek characters: \(\alpha\), \(\beta\), \(\lambda\).

#### Vector

An ordered tuple of scalars from \(\mathbb{F}\). An \(n\)-dimensional real vector is an element of the space \(\mathbb{R}^n\):

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

{{< callout type="info" >}}
In plain English: \(\mathbf{v}\) is a column of \(n\) real numbers. The subscript tells you which *"slot"* you're in.
{{< /callout >}}

#### Espacio vectorial

A set \(V\) of vectors over a field \(\mathbb{F}\), equipped with two operations, for which it is ***closed***.

**Vector addition**: \(\mathbf{u} + \mathbf{v} \in V\) for all \(\mathbf{u}, \mathbf{v} \in V\). Internal operation with the following properties:
- *Associativity*: \(\mathbf{u} + (\mathbf{v} + \mathbf{w}) = (\mathbf{u} + \mathbf{v}) + \mathbf{w}\).
- *Commutativity*: \(\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}\).
- *Identity element*: there exists a vector \(\mathbf{0} \in V\), called zero vector, such that \(\mathbf{v} + \mathbf{0} = \mathbf{v}\) for all \(\mathbf{v} \in V\).
- *Inverse element*: for all \(\mathbf{v} \in V\), there exist an element \(\mathbf{-v} \in V\) such as \(\mathbf{v} + (\mathbf{-v}) = \mathbf{0}\).

**Scalar multiplication**: \(\alpha \mathbf{v} \in V\) for all \(\alpha \in \mathbb{F}, \mathbf{v} \in V\).  External operation with the following properties:
- *Associativity*: \(\alpha (\beta \mathbf{v}) = (\alpha \beta) \mathbf{v}\).
- *Identity element*: there exists an scalar \(\alpha\), such as \(\alpha \mathbf{v} = \mathbf{v} \alpha = \mathbf{v}\) para todo \(\mathbf{v} \in V\).
- *Distributivity of scalar multiplication with respect to vector addition*: For any scalar \(\alpha\), it is true that \(\alpha (\mathbf{u} + \mathbf{v}) = \alpha \mathbf{u} + \alpha \mathbf{v}\) for all \(\mathbf{u}, \mathbf{v} \in V\).
- *Distributivity of scalar multiplication with respect to scalars addition*: For any two scalars \(\alpha\) and \(\beta\), it is true that \((\alpha + \beta) \mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}\) for every \(\mathbf{v} \in V\).

{{< callout >}}
What does it mean that the vector space is closed for those operations?

It means that they produce results that are in the same vector space.
- If \(\mathbf{u} \in V\) and \(\mathbf{v} \in V\) then \((\mathbf{u} + \mathbf{v}) \in V\).
- For any scalar \(\alpha \in \mathbb{R}\) and vector \(\mathbf{v} \in V\), then \(\alpha \mathbf{v} \in V\).
{{< /callout >}}

### Vector addition

Given \(\mathbf{u} = [u_1, u_2, \ldots, u_n]^T\) and \(\mathbf{v} = [v_1, v_2, \ldots, v_n]^T\):

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

{{< callout type="info" >}}
In plain English: add element-by-element. Geometrically, place the tail of \(\mathbf{v}\) at the head of \(\mathbf{u}\), the result is the arrow from start to finish following the [parallelogram law](https://en.wikipedia.org/wiki/Parallelogram_law).
{{< /callout >}}

### Vector norms

The **norm** of a vector measures its length. The most common is the **Euclidean norm** (\(L^2\) norm):

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

{{< callout type="info" >}}
In plain English: square each component, sum them, take the square root. This is precisely the Pythagorean theorem generalized to \(n\) dimensions.
{{< /callout >}}

The general family is the **\(L^p\) norm**:

$$
\|\mathbf{v}\|_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}
$$

Two special cases appear constantly in Machine Learning (ML):

- **\(L^1\) norm (Manhattan)**:

    Used in LASSO regularization because it induces *sparsity*, it penalizes any nonzero component equally.
    $$
    \|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|
    $$

- **\(L^\infty\) norm (max norm)**:

    Useful when you care about the single largest activation.
    $$
    \|\mathbf{v}\|_\infty = \max_i |v_i|
    $$

To dig more about Vector norms, check [this](https://en.wikipedia.org/wiki/Norm_(mathematics)) Wikipedia article.

### The dot product

The **dot product** (inner product) of two vectors is:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n
$$

{{< callout type="info" >}}
In plain English: multiply corresponding components and sum the results. The output is a scalar, a single number that encodes how much the two vectors ***"align"***.
{{< /callout >}}

{{< callout >}}
The dot product of a vector and itself results in its magnitude squared.
$$
\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2
$$
{{< /callout >}}

### The angle between vectors

Here is where geometry and algebra merge beautifully. From Euclidean geometry, the [**Law of Cosines**](https://en.wikipedia.org/wiki/Law_of_cosines) states that for a triangle with sides \(a\), \(b\), \(c\) and angle \(\theta\) opposite side \(c\):

$$
c^2 = a^2 + b^2 - 2 a b \cos(\theta)
$$

Now apply this to vectors. Let \(\mathbf{u}\) and \(\mathbf{v}\) be two vectors. The third side of the triangle they form is \(\mathbf{u} - \mathbf{v}\).

Substituting \(a = \|\mathbf{u}\|\), \(b = \|\mathbf{v}\|\), \(c = \|\mathbf{u} - \mathbf{v}\|\):

$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\|\ cos(\theta)
$$

Expanding the left side:

$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

$$
(\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

$$
\|\mathbf{u}\|^2 - 2 (\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Canceling \(\|\mathbf{u}\|^2\) and \(\|\mathbf{v}\|^2\) from both sides:
$$
-2 (\mathbf{u} \cdot \mathbf{v}) = -2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Dividing both sides by \(-2 \|\mathbf{u}\| \|\mathbf{v}\|\) (assuming neither vector is zero):
$$
\frac{-2 (\mathbf{u} \cdot \mathbf{v})}{-2 \|\mathbf{u}\| \|\mathbf{v}\|} = \frac{-2 \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)}{-2 \|\mathbf{u}\| \|\mathbf{v}\|}
$$

$$
\boxed{\frac{(\mathbf{u} \cdot \mathbf{v})}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos(\theta)}
$$

{{< callout type="info" >}}
In plain English: the cosine of the angle between two vectors equals their dot product divided by the product of their lengths. This formula is ***foundational***, it gives us **cosine similarity**, one of the most ubiquitous distance metrics in Machine Learning.
{{< /callout >}}

Key interpretations:
- \(\cos\theta = 1\) (\(\theta = 0°\)): vectors point in the **same direction** (identical topics in a document embedding).
- \(\cos\theta = 0\) (\(\theta = 90°\)): vectors are **orthogonal**, completely unrelated.
- \(\cos\theta = -1\) (\(\theta = 180°\)): vectors point in **opposite directions** (antonyms in a well-trained embedding space).

{{< callout >}}
In the original [Word2Vec](https://arxiv.org/abs/1301.3781) paper (Mikolov et al., 2013), the famous analogy:
$$
king − man + woman ≈ queen
$$
works precisely because of this geometry. Semantic relationships are encoded as *directions* in vector space, and finding `queen` means finding the vector whose cosine similarity to the query vector is maximized. Every modern embedding model (BERT, GPT, sentence-transformers) inherits this geometric philosophy. Next time you read something about word representation in vector spaces, remember they are talking about the same geometry we just derived.
{{< /callout >}}
