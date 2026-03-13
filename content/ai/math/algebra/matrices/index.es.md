---
weight: 2
title: "Matrices: Operaciones y Propiedades"
authors:
  - jnonino
description: >
  Domina la aritmética matricial, transformaciones lineales, determinantes, rango e inversas desde cero. Derivaciones matemáticas paso a paso, código Python ejecutable e insights de nivel investigación para aspirantes a Científicos de ML.
date: 2026-03-13
tags: ["IA", "Matemáticas", "Álgebra", "Álgebra Lineal", "Matrices"]
---

GPT-3 almacena su *"inteligencia"* en aproximadamente 1800 matrices de pesos. Cada *token* que escribís dispara una cascada de multiplicaciones a través de esas matrices (proyecciones, puntuaciones de atención, expansiones feed-forward) antes de que se produzca una única probabilidad de salida. El modelo completo de 175.000 millones de parámetros es, en su núcleo computacional, una composición intrincada de operaciones matriciales aplicadas en secuencia.

Las matrices hacen que la idea abstracta de un espacio vectorial se convierta en un motor que podés ejecutar de verdad. Un vector te dice dónde está algo; una matriz te dice cómo moverlo. Las matrices de pesos en las redes neuronales codifican transformaciones aprendidas del espacio de características. Las matrices de covarianza en estadística codifican la forma de una distribución de datos. La matriz Jacobiana de una función codifica cómo cambian sus salidas respecto a sus entradas, es el objeto que hace posible la *retropropagación*.

El problema es que la mayoría de las introducciones a las matrices las tratan como planillas de cálculo glorificadas: grillas de números con reglas aritméticas pegadas encima. Esa perspectiva hace que las operaciones parezcan arbitrarias. Este artículo toma el enfoque opuesto: arrancamos desde la geometría, derivamos cada operación desde primeros principios, y al final vas a ver las matrices no como tablas sino como **funciones**, con toda la riqueza que eso implica: composición, invertibilidad, imagen, núcleo y rango.

Al finalizar este artículo, serás capaz de:

- Definir formalmente una matriz y una transformación lineal, y explicar por qué importan los axiomas.
- Ejecutar y derivar todas las operaciones matriciales fundamentales: suma, multiplicación, transpuesta, traza e inversión.
- Calcular el determinante de manera geométrica y algebraica, y entender qué significa un determinante nulo.
- Razonar sobre el espacio columna, espacio fila, espacio nulo y rango. Enunciar y aplicar el Teorema Rango-Nulidad.
- Implementar estas operaciones desde cero en Python puro y NumPy.
- Conectar estas operaciones con la investigación contemporánea en aprendizaje profundo (*deep learning*).

Comencemos.

## Prerrequisitos

Antes de leer este artículo, deberías dominar:

- **Vectores y espacios vectoriales**: definiciones formales, productos punto, normas, espacio generado y combinaciones lineales.
- **Python y NumPy básico**: creación de arreglos, indexación, manipulación de formas (*shapes*).
- **Notación de funciones**: entender notación del estilo \(f: \mathbb{R}^m \rightarrow \mathbb{R}^n\).

Si puedes definir qué significa que un conjunto sea *cerrado* bajo una operación, tienes la base suficiente.
