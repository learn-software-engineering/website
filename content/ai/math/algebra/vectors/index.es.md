---
weight: 1
title: "Vectores, Escalares y Espacios Vectoriales"
authors:
  - jnonino
description: >
  Domina vectores y espacios vectoriales para machine learning: intuición geométrica, producto punto, similitud coseno, normas, código NumPy y perspectiva de investigación para científicos en Machine Learning (ML).
date: 2026-03-06
tags: ["AI", "Maths", "Algebra", "Vectors"]
---

En 2017, investigadores de Google Brain publicaron [**Attention Is All You Need**](), el artículo que introdujo la arquitectura Transformer sobre la que hoy se construyen GPT-4, Gemini y prácticamente todos los modelos de lenguaje de vanguardia. En el corazón de esa arquitectura (y de toda red neuronal, sistema de recomendación y modelo de visión por computadora—) vive un objeto engañosamente simple: el **vector**.

Cuando un modelo de lenguaje lee la palabra *"banco"*, no ve una cadena de texto. Ve un vector en un espacio de 4096 dimensiones donde *"banco (financiero)"* y *"banco (asiento)"* ocupan regiones mensurablemente distintas. Cuando un motor de búsqueda decide que tu consulta coincide con un documento, está calculando el ángulo entre dos vectores. Cuando una red neuronal aprende, está desplazando vectores por el espacio en respuesta a un gradiente, que es, él mismo, otro vector.

Este artículo construye tu fundamento operativo para todo eso. Al terminar de leerlo, podrás:

- Definir formalmente vectores, escalares y espacios vectoriales, y explicar *por qué* importan los axiomas.
- Calcular normas, productos punto y ángulos entre vectores a mano y con *NumPy*.
- Razonar geométricamente sobre datos de alta dimensión, una habilidad indispensable en investigación en Machine Learning.
- Leer un artículo de investigación con notación vectorial sin perder el hilo.

Sin mas preámbulos, empecemos.

## Prerrequisitos

Antes de leer este artículo, deberías manejar con comodidad:

- **Álgebra de secundaria**: variables, funciones, el plano cartesiano.
- **Python básico**: listas, bucles, funciones, importación de bibliotecas.
- **Intuición de cálculo** (útil, no obligatorio): la idea de que una derivada apunta en la dirección de mayor pendiente.

## Intuición primero

### La analogía del programador: vectores como arreglos tipados con alma geométrica

Como desarrollador, has usado arreglos toda tu carrera. Una lista de Python `[3.0, -1.5, 7.2]` almacena tres números. Un vector es superficialmente lo mismo, pero con una estructura adicional crucial: **posición en el espacio y la geometría que conecta posiciones**.

Considera este ejemplo. Supón que tienes dos diccionarios que representan usuarios de una plataforma de comercio electrónico:

```python
usuario_A = {"edad": 28, "freq_compras": 5, "gasto_promedio": 1200.0}
usuario_B = {"edad": 29, "freq_compras": 6, "gasto_promedio": 1150.0}
```

Como diccionarios, son solo contenedores de datos. Puedes leer valores, pero *"¿qué tan similares son estos usuarios?"* no es una pregunta que el diccionario pueda responder de forma nativa. Ahora conviértelos en vectores:

```python
A = [28, 5, 1200.0]
B = [29, 6, 1150.0]
```

De repente tienes geometría. Puedes medir la *distancia* entre ellos, el *ángulo* que forman respecto al origen, y si uno es una *versión escalada* del otro. Este es el salto que dan los vectores sobre los arreglos simples: **viven en un espacio equipado con reglas para medir, comparar y transformar**.

### El diagrama conceptual: vectores como flechas

Imagina un sistema de coordenadas 2D estándar. El vector \(\mathbf{v} = [3, 2]\) es una flecha que parte del origen \((0, 0)\) y termina en el punto \((3, 2)\). Dos cosas lo definen completamente: su **magnitud** (qué tan larga es la flecha) y su **dirección** (hacia dónde apunta).

{{< figure
    src="images/vector.png"
    alt="Representación visual de un vector en dos dimensiones"
    caption="Representación visual de un vector en dos dimensiones"
    >}}

Esta interpretación geométrica no es solo azúcar visual. En aprendizaje automático, un punto de datos (una fila en tu dataset) *es* un vector, es una flecha en el espacio de características. Dos puntos similares son flechas apuntando en direcciones parecidas. Un valor atípico es una flecha apuntando hacia algún lugar inesperado. La reducción de dimensionalidad (PCA, UMAP) es el arte de encontrar un espacio de menor dimensión donde esas flechas cuenten esencialmente la misma historia.

### Escalares: el caso más simple

Un **escalar** es simplemente un número solo, sin dirección, sin componentes. La temperatura, el valor de pérdida (*loss*), la tasa de aprendizaje: todos son escalares. Cuando multiplicas un vector por un escalar, estiras o encoges la flecha sin rotarla:

Misma dirección, el doble de largo.
$$
2 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 3 \\ 2 \cdot 2 \end{pmatrix} = \begin{pmatrix} 6 \\ 4 \end{pmatrix}
$$
```python
2 x [3, 2] = [6, 4]
```

Misma dirección, invertida (180°).
$$
-1 \cdot \begin{pmatrix} 3 \\ 2 \end{pmatrix} = \begin{pmatrix} -1 \cdot 3 \\ -1 \cdot 2 \end{pmatrix} = \begin{pmatrix} -3 \\ -2 \end{pmatrix}
$$
```python
-1 x [3, 2] = [-3, -2]
```

Esta operación, **multiplicación escalar**, es una de las dos operaciones fundamentales que definen un espacio vectorial.

{{< callout type="important" >}}
Cuando depures una red neuronal cuya pérdida explota (*exploding gradients*), frecuentemente significa que vectores de activaciones o gradientes se están escalando por factores mayores a 1.0 en cada capa. Entender la multiplicación escalar geométricamente te ayuda a ver *por qué* el recorte de gradientes (*gradient clipping*) o la normalización por lotes (*batch normalization*) restauran la estabilidad: están renormalizando la longitud de esas flechas hacia valores manejables.
{{< /callout >}}

## Derivación matemática
