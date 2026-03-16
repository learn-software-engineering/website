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

### Matrices as linear transformations

The most important property of a matrix is that it defines a **linear transformation** \(T: \mathbb{R}^n \rightarrow \mathbb{R}^m\) via \(T(\mathbf{x}) = \mathbf{A}\mathbf{x}\).

A function \(T\) is called **linear** if and only if it satisfies two axioms for all vectors \(\mathbf{u}, \mathbf{v} \in \mathbb{R}^n\) and all scalars \(\alpha \in \mathbb{R}\):

- *Additivity*: \(T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})\)
- *Homogeneity*: \(T(\alpha \mathbf{u}) = \alpha T(\mathbf{u})\)

These two axioms are equivalent to the single condition:

$$
T(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})
$$

{{< callout type="info" >}}
In plain English: a linear transformation preserves the structure of the vector space, it doesn't matter whether you add two vectors first and then transform, or transform each separately and add. This is why stacking multiple linear (un-activated) layers in a neural network collapses to a single matrix product: \(\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}\).
{{< /callout >}}

A foundational theorem from linear algebra states that **every** linear transformation between finite-dimensional spaces can be represented as a matrix, and conversely every matrix defines a linear transformation. The matrix and the linear transformation are, for our purposes, the same object.

### Matrix addition and scalar multiplication

The set \(\mathbb{R}^{m \times n}\) of all \(m \times n\) real matrices forms a **vector space** under the following operations.

**Matrix addition**: Given \(\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}\):

$$
(\mathbf{A} + \mathbf{B})_{ij} = A_{ij} + B_{ij}
$$

This satisfies:
- *Commutativity*: \(\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}\)
- *Associativity*: \((\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})\)
- *Identity element*: there exists a zero matrix \(\mathbf{0}\) such that \(\mathbf{A} + \mathbf{0} = \mathbf{A}\)
- *Inverse element*: for each \(\mathbf{A}\), there exists \(-\mathbf{A}\) such that \(\mathbf{A} + (-\mathbf{A}) = \mathbf{0}\)

**Scalar multiplication**: Given \(\alpha \in \mathbb{R}\) and \(\mathbf{A} \in \mathbb{R}^{m \times n}\):

$$(\alpha \mathbf{A})_{ij} = \alpha \cdot A_{ij}$$

This satisfies:
- *Associativity*: \(\alpha(\beta \mathbf{A}) = (\alpha\beta)\mathbf{A}\)
- *Identity element*: \(1 \cdot \mathbf{A} = \mathbf{A}\)
- *Distributivity over matrix addition*: \(\alpha(\mathbf{A} + \mathbf{B}) = \alpha\mathbf{A} + \alpha\mathbf{B}\)
- *Distributivity over scalar addition*: \((\alpha + \beta)\mathbf{A} = \alpha\mathbf{A} + \beta\mathbf{A}\)

{{< callout >}}
If these axioms look familiar, they should. They are exactly the vector space axioms from our previous lesson about vectors. Matrices *"are"* vectors in a higher-dimensional space. A \(3 \times 4\) matrix is just a vector with \(12\) components arranged in a grid. This means every theorem we prove about vector spaces applies to matrices too.
{{< /callout >}}

### Matrix multiplication

Given \(\mathbf{A} \in \mathbb{R}^{m \times k}\) and \(\mathbf{B} \in \mathbb{R}^{k \times n}\), their product \(\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}\) is defined as:

$$\boxed{C_{ij} = \sum_{l=1}^{k} A_{il} \cdot B_{lj}}$$

To compute entry \(C_{ij}\), the element in row \(i\), column \(j\) of the result, take the **dot product of row \(i\) of \(\mathbf{A}\) with column \(j\) of \(\mathbf{B}\)**. The inner dimensions must match:

$$
\underbrace{\mathbf{A}}_{m \times k} \cdot \underbrace{\mathbf{B}}_{k \times n} = \underbrace{\mathbf{C}}_{m \times n}
$$

{{< callout type="info" >}}
In plain English: matrix multiplication is function composition. Applying transformation \(\mathbf{B}\) first, then \(\mathbf{A}\), gives the same result as the single composed transformation \(\mathbf{AB}\). The inner dimensions must agree because the output dimension of the first transformation must equal the input dimension of the second, exactly like composing functions.
{{< /callout >}}

Matrix multiplication satisfies:
- *Associativity*: \((\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})\)
- *Left distributivity*: \(\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}\)
- *Right distributivity*: \((\mathbf{A} + \mathbf{B})\mathbf{C} = \mathbf{AC} + \mathbf{BC}\)
- *Identity*: \(\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}\) (for compatible \(\mathbf{I}\))
- **Non-commutativity**: \(\mathbf{AB} \neq \mathbf{BA}\) in general

{{< callout >}}
The non-commutativity is not a flaw but a fundamental feature of transformations. Rotating a figure then reflecting it is geometrically different from reflecting then rotating.
{{< /callout >}}

### The transpose

The **transpose** of \(\mathbf{A} \in \mathbb{R}^{m \times n}\) is the matrix \(\mathbf{A}^\top \in \mathbb{R}^{n \times m}\) defined by:

$$
\boxed{(\mathbf{A}^\top)_{ij} = A_{ji}}
$$

{{< callout type="info" >}}
In plain English: flip the matrix along its main diagonal, rows become columns and columns become rows. A \(3 \times 5\) matrix becomes a \(5 \times 3\) matrix. Geometrically, the transpose corresponds to the *adjoint* of the transformation, the transformation that "undoes the rotation part" while keeping the scaling.
{{< /callout >}}

The transpose satisfies the following properties:

- *Double transpose*: \((\mathbf{A}^\top)^\top = \mathbf{A}\)
- *Linearity*: \((\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top\) and \((\alpha\mathbf{A})^\top = \alpha\mathbf{A}^\top\)
- *Product reversal*: \((\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top\)

The product reversal property is **critical** and appears in every backpropagation derivation. Let us prove it completely. We want to show \([(\mathbf{AB})^\top]_{ij} = [\mathbf{B}^\top \mathbf{A}^\top]_{ij}\).

Starting from the definition of transpose:

$$
[(\mathbf{AB})^\top]_{ij} = [\mathbf{AB}]_{ji}
$$

Applying the definition of matrix multiplication to \([\mathbf{AB}]_{ji}\):

$$
[\mathbf{AB}]_{ji} = \sum_{l} A_{jl} \cdot B_{li}
$$

Recognising each factor using the definition of transpose (\(A_{jl} = [\mathbf{A}^\top]_{lj}\) and \(B_{li} = [\mathbf{B}^\top]_{il}\)):

$$\sum_{l} A_{jl} B_{li} = \sum_{l} [\mathbf{B}^\top]_{il} \cdot [\mathbf{A}^\top]_{lj}$$

Recognising the right-hand side as the definition of matrix multiplication \( [\mathbf{B}^\top \mathbf{A}^\top]_{ij} \):

$$
\boxed{(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top}
$$

**Symmetric and skew-symmetric matrices** are important special cases:

- \(\mathbf{A}\) is **symmetric** if \(\mathbf{A}^\top = \mathbf{A}\), equivalently \(A_{ij} = A_{ji}\) for all \(i, j\). Covariance matrices and kernel matrices in ML are always symmetric.
- \(\mathbf{A}\) is **skew-symmetric** if \(\mathbf{A}^\top = -\mathbf{A}\), equivalently \(A_{ij} = -A_{ji}\) and all diagonal entries are zero.

{{< callout >}}
Any square matrix \(\mathbf{A}\) can be uniquely decomposed into a symmetric part and a skew-symmetric part:

$$
\mathbf{A} = \underbrace{\frac{\mathbf{A} + \mathbf{A}^\top}{2}}_{\text{symmetric}} + \underbrace{\frac{\mathbf{A} - \mathbf{A}^\top}{2}}_{\text{skew-symmetric}}
$$

This decomposition has applications in physics (strain vs. rotation analysis) and in recent research on attention mechanism structure.
{{< /callout >}}

### The trace

For a square matrix \(\mathbf{A} \in \mathbb{R}^{n \times n}\), the **trace** is the sum of its diagonal entries:

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^{n} A_{ii}
$$

{{< callout type="info" >}}
In plain English: add up the main diagonal. The trace captures how much the transformation *"expands"* space on average.
{{< /callout >}}

Key properties:
- *Linearity*: \(\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})\) and \(\text{tr}(\alpha\mathbf{A}) = \alpha\,\text{tr}(\mathbf{A})\)
- *Transpose invariance*: \(\text{tr}(\mathbf{A}^\top) = \text{tr}(\mathbf{A})\)
- **Cyclic property**: \(\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB})\)

The cyclic property is everywhere in gradient derivations. Let us prove the base case \(\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})\).

Starting from the definition of trace:

$$\text{tr}(\mathbf{AB}) = \sum_{i} [\mathbf{AB}]_{ii} = \sum_{i} \sum_{j} A_{ij} B_{ji}$$

Swapping the order of summation (valid since both sums are finite):

$$\sum_{i} \sum_{j} A_{ij} B_{ji} = \sum_{j} \sum_{i} B_{ji} A_{ij}$$

Recognising the right-hand side as the diagonal entry \([\mathbf{BA}]_{jj}\):

$$
\sum_{j} \sum_{i} B_{ji} A_{ij} = \sum_{j} [\mathbf{BA}]_{jj} = \text{tr}(\mathbf{BA})
$$

$$
\boxed{\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})}
$$

{{< callout >}}
Note that this holds even when \(\mathbf{AB}\) and \(\mathbf{BA}\) have different shapes (e.g., \(\mathbf{A} \in \mathbb{R}^{m \times n}\), \(\mathbf{B} \in \mathbb{R}^{n \times m}\)), both traces are scalar and equal.
{{< /callout >}}
