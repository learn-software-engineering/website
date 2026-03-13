---
weight: 2
title: "Matrices: Operations and Properties"
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
