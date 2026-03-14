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

## Intuición primero

### La analogía del programador: matrices como funciones

Como desarrollador, has usado funciones toda tu carrera. Una función `transformar(x)` recibe una entrada y la mapea a una salida. Una matriz \(\mathbf{A}\) es exactamente eso, una **función** que toma un vector como entrada y produce un vector como salida. La restricción clave es que esta función debe ser *lineal*, lo que impone una estructura geométrica específica sobre qué transformaciones están permitidas.

Pensalo de esta manera. Imaginate un pipeline de datos donde vectores de características de usuarios pasan por etapas de procesamiento:

```python
# Etapa 1: expandir 3 características crudas a 5 características derivadas
etapa1 = transformar_3_a_5(caracteristicas_usuario)   # shape: (5,)

# Etapa 2: comprimir 5 características derivadas a 2 dimensiones latentes
etapa2 = transformar_5_a_2(etapa1)                    # shape: (2,)

# Combinado: ¿podemos hacer ambas en un solo paso?
combinado = transformar_3_a_2(caracteristicas_usuario)  # Sí, multiplicación de matrices
```

Esto es exactamente lo que calcula la multiplicación de matrices: la **composición** de dos transformaciones lineales en una. Una matriz \(5 \times 3\) (etapa 1) compuesta con una matriz \(2 \times 5\) (etapa 2) produce una única matriz \(2 \times 3\) que hace ambos pasos a la vez. Cada capa de una red neuronal es una etapa de este pipeline.

### La imagen geométrica: matrices como transformaciones del espacio

Imagina un sistema de coordenadas 2D. Los vectores de la [base canónica](https://es.wikipedia.org/wiki/Base_can%C3%B3nica) son:

$$
\hat{e}_1 = [1,0] \qquad \text{(apunta a la derecha a lo largo del eje x)}
$$

$$
\hat{e}_2 = [0,1] \qquad \text{(apunta hacia arriba a lo largo del eje y)}
$$

Aplica la matriz \(\mathbf{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}\). Tras la transformación: \(\hat{e}_1\) se mapea a \([2, 0]\) (estirado a la derecha por factor 2) y \(\hat{e}_2\) se mapea a \([0, 3]\) (estirado hacia arriba por factor 3). El cuadrado unitario (con área 1), se convierte en un rectángulo de \(2 \times 3\) con área 6. El **determinante** de \(\mathbf{A}\), que derivaremos en breve, es exactamente \(6\). Esto no es una coincidencia: el determinante *es* el factor de escalado del volumen.

Compara esto con una matriz de rotación:

$$
\mathbf{R}(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

Ésta rota cada vector por el ángulo \(\theta\) sin estirarlo, por lo que preserva todas las áreas, y su determinante siempre es 1.

Las columnas de cualquier matriz te dicen exactamente adónde van los vectores de la base. Dado que todo vector es una combinación lineal de los vectores base, saber adónde van los vectores base te dice adónde va *cada* vector del espacio. Este razonamiento explica por qué la multiplicación de matrices no es conmutativa (aplicar la transformación A y luego B es geométricamente diferente de B y luego A), y por qué las columnas de las matrices de pesos en las redes neuronales tienen un significado semántico que los investigadores analizan activamente.

{{< callout type="important" >}}
Las columnas de una matriz no son sólo números, son las *imágenes de los vectores de la base* bajo la transformación. Cuando examinas la matriz de pesos \(\mathbf{W}\) de una capa de red neuronal entrenada, cada columna te dice cómo responde esa capa a una dirección de entrada estándar. Éste es el fundamento de la investigación de visualización de características, donde los practicantes interpretan qué *"detecta"* cada neurona examinando las direcciones en las matrices de pesos.
{{< /callout >}}
