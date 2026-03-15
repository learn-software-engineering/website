---
weight: 2
title: "Matrices: operations and properties"
authors:
  - jnonino
description: >
  Master matrix arithmetic, linear transformations, determinants, rank, and inverses from scratch. Step-by-step math derivations, runnable Python code, and research-grade insights for aspiring Machine Learning (ML) scientists.
date: 2026-03-13
tags: ["AI", "Maths", "Algebra", "Linear Algebra", "Matrices"]
---

GPT-3 stores its *"intelligence"* in approximately 1800 weight matrices. Every token you type triggers a cascade of multiplications across those matrices (projections, attention scores, feed-forward expansions) before a single output probability is produced. The entire 175-billion-parameter model is, at its computational core, an intricate composition of matrix operations applied in sequence.

Matrices are where the abstract idea of a vector space becomes an engine you can actually run. A vector tells you where something is; a matrix tells you how to move it. Weight matrices in neural networks encode learned transformations of feature space. Covariance matrices in statistics encode the shape of a data distribution. The Jacobian matrix of a function encodes how its outputs change with respect to its inputs, it is the object that makes *backpropagation* possible.

The trouble is that most introductions to matrices treat them as glorified spreadsheets: grids of numbers with arithmetic rules bolted on. That framing makes the operations feel arbitrary. This article takes the opposite approach: we start from the geometry, derive every operation from first principles, and by the end you will see matrices not as tables but as **functions**, with all the richness that implies: composition, invertibility, image, kernel, and rank.

By the end of this article, you will be able to:

- Formally define a matrix and a linear transformation, and explain why the axioms matter.
- Execute and derive all fundamental matrix operations: addition, multiplication, transpose, trace, and inversion.
- Compute the determinant geometrically and algebraically, and understand what a zero determinant means.
- Reason about column space, row space, null space, and rank. And state and apply the Rank-Nullity Theorem.
- Implement these operations from scratch in pure Python and NumPy.
- Connect these operations to contemporary research in deep learning.

Let's start.

## Prerequisites

Before reading this article, you should be comfortable with:

- **Vectors and vector spaces**: formal definitions, dot products, norms, span, and linear combinations.
- **Basic Python and NumPy**: array creation, indexing, shape manipulation.
- **Function notation**: understanding \(f: \mathbb{R}^m \rightarrow \mathbb{R}^n\) style notation.

If you can define what it means for a set to be *closed* under an operation, you have enough scaffolding.

## Intuition first

### The programmer's analogy: matrices as functions

As a developer, you have used functions your whole career. A function `transform(x)` takes an input and maps it to an output. A matrix \(\mathbf{A}\) is exactly that, a **function** that takes a vector as input and produces a vector as output. The key constraint is that this function must be *linear*, which imposes a specific geometric structure on what transformations are permitted.

Think of it this way. Imagine a data pipeline where user feature vectors pass through processing stages:

```python
# Stage 1: expand 3 raw features into 5 derived features
stage1 = transform_3_to_5(user_features)   # shape: (5,)

# Stage 2: compress 5 derived features into 2 latent dimensions
stage2 = transform_5_to_2(stage1)           # shape: (2,)

# Combined: can we do both in one step?
combined = transform_3_to_2(user_features)  # Yes — matrix multiplication
```

This is exactly what matrix multiplication computes: the **composition** of two linear transformations into one. A \(5 \times 3\) matrix (stage 1) composed with a \(2 \times 5\) matrix (stage 2) produces a single \(2 \times 3\) matrix that does both steps at once. Every layer of a neural network is one stage of this pipeline.

### Geometric picture: matrices as space transformations

Picture a 2D coordinate system. The [standard basis](https://en.wikipedia.org/wiki/Standard_basis) vectors are:

$$
\hat{e}_1 = [1,0] \qquad \text{(points right along x-axis)}
$$

$$
\hat{e}_2 = [0,1] \qquad \text{(points up along y-axis)}
$$

Apply the matrix \(\mathbf{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}\). After the transformation: \(\hat{e}_1\) maps to \([2,0]\) (stretched right by factor 2) and \(\hat{e}_2\) maps to \([0,3]\) (stretched up by factor 3). The unit square (with area 1), becomes a \(2 \times 3\) rectangle with area 6. The **determinant** of \(\mathbf{A}\), which we will derive shortly, equals exactly 6. This is not a coincidence: the determinant *is* the volume scaling factor.

Compare this to a rotation matrix:

$$
\mathbf{R}(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

This rotates every vector by angle $\theta$ without stretching, so it preserves all areas — and its determinant is always $1$.

The columns of any matrix tell you exactly where the basis vectors land. Since every vector is a linear combination of the basis vectors, knowing where basis vectors go tells you where *every* vector in the space goes. This insight explains why matrix multiplication is non-commutative (applying transformation A then B is geometrically different from B then A), and why the columns of weight matrices in neural networks carry semantic meaning that researchers actively analyse.

{{< callout type="important" >}}
The columns of a matrix are not just numbers, they are the *images of the basis vectors* under the transformation. When you look at the weight matrix \(\mathbf{W}\) of a trained neural network layer, each column tells you how that layer responds to one standard input direction. This is the foundation of feature visualization research, where practitioners interpret what each neuron *"detects*" by examining the directions in weight matrices.
{{< /callout >}}

## Mathematical derivation

### Formal definition

An **\(m \times n\) matrix** over \(\mathbb{R}\) is a rectangular array of real numbers arranged in \(m\) rows and \(n\) columns:

$$
\mathbf{A} = \begin{bmatrix} A_{11} & A_{12} & \cdots & A_{1n} \\ A_{21} & A_{22} & \cdots & A_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ A_{m1} & A_{m2} & \cdots & A_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}
$$

The entry \(A_{ij}\) denotes the element in **row \(i\), column \(j\)**. We write the set of all \(m \times n\) real matrices as \(\mathbb{R}^{m \times n}\).

{{< callout type="info" >}}
In plain English: a matrix is a 2D table of numbers. The notation \(\mathbb{R}^{m \times n}\) is the *"type"* of the matrix (how many rows and columns it has). This is exactly the type signature you would attach to a 2D array in a statically typed language: `Array[Float, m, n]`.
{{< /callout >}}

**Special cases** you will encounter constantly:

- **Column vector**: a matrix of shape \(n \times 1\), this is just a vector \(mathbf{v} \in \mathbb{R}^n\).
- **Row vector**: a matrix of shape \(1 \times n\).
- **Square matrix**: a matrix where \(m = n\).
- **Identity matrix** \(\mathbf{I}_n\): the \(n \times n\) matrix with \(I_{ij} = 1\) if \(i = j\), else \(0\).
- **Zero matrix** \(\mathbf{0}\): all entries are zero.
- **Diagonal matrix**: a square matrix where \(A_{ij} = 0\) for all \(i \neq j\).
