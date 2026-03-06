---
weight: 1
title: "Arreglos"
cardImage: "featured.es.jpg"
authors:
  - jnonino
description: >
  Los arreglos (arrays en inglés) son estructuras de datos fundamentales en programación que permiten almacenar y organizar colecciones de datos del mismo tipo. Dominar el uso de arrays es esencial para cualquier programador.
date: 2023-10-30
tags: ["Programación", "Estructuras de Datos", "Arreglos"]
---

Un arreglo es una estructura de datos que representa un conjunto de elementos, los cuales se acceden a través de índices numéricos contiguos que van desde 0 hasta el tamaño del array menos 1. Proveen acceso rápido y directo a los elementos en base a su posición.

En lenguajes como Python y Ruby, los arrays se conocen como *listas* (lists). En Javascript se les conoce como *arreglos* (arrays).

Son típicamente homogéneos, almacenando elementos del mismo tipo como enteros, cadenas, etc. Algunos lenguajes permiten arrays heterogéneos con valores de distintos tipos.

{{< figure
    src="images/arrays.jpg"
    alt="Diagrama de un array"
    caption="Diagrama de un array"
    >}}

### Creación

La manera de crear arreglos varía según el lenguaje de programación:

```python
ARREGLO_VACIO = []
MI_ARREGLO = ["A", "B", "C"] # Arreglo literal
mi_arreglo = list(range(5))  # Arreglo a partir de rango
```

Al crear un arreglo literal se inicializan sus elementos directamente. Al construir uno vacío, en algunos lenguajes  se especifica su tamaño, pero sus elementos son inicializados con un valor default (0 para números, null para objetos, etc).

### Acceder y modificar elementos

Los elementos individuales se acceden rápidamente por su índice utilizando corchetes `[]`:

```python
my_arreglo = ['a', 'b', 'c']
print(my_arreglo[0]) # 'a'
print(my_arreglo[2]) # 'c'
my_arreglo[2] = 'z'
print(my_arreglo[2]) # 'z'
```

Los índices comienzan en 0, por lo que en un array de tamaño \(n\), los índices válidos están entre 0 y \(n - 1\).

Acceder a un índice inválido causa un error, por ejemplo, acceder al índice 3 en un array de tamaño 3. Esto se conoce como `index out of bounds`.

### Recorrer un arreglo

Podemos recorrer todos los elementos usando un ciclo `for`:

```python
letras = ['a', 'b', 'c']

for i in range(len(letras)):
    print(letras[i])
```

Esto imprime cada elemento en orden. `len()` devuelve la longitud total del array.

Otra forma es iterando directamente sobre los elementos:

```python
letras = ['a', 'b', 'c']

for letra in letras:
    print(letra)
```

## Buscar en un arreglo

Podemos buscar un elemento en un array mediante un ciclo y comparando elemento por elemento:

```python
letras = ['a', 'b', 'c']

def buscar_en_arreglo(arreglo, elemento):
    for i in range(len(arreglo)):
        if arreglo[i] == elemento:
            return i
    return False

print(buscar_en_arreglo(letras, 'b')) # 1
print(buscar_en_arreglo(letras, 'z')) # False
```

Devuelve el índice si se encuentra o `False` si no se encuentra.

## Arreglo multidimensional

Los arrays pueden tener más de una dimensión, por ejemplo matrices 2D, cubos 3D, etc.

En 2 dimensiones (2D) se puede ver como una tabla con filas y columnas. Para acceder a un elemento se especifican dos índices, uno para la fila y otro para la columna:

```python
matrix = [
  [1, 2, 3],
  [4, 5, 6]
]

print(matrix[0][2]) # 3
print(matrix[1][1]) # 5
```

Pueden tener más dimensiones, por ejemplo un arreglo 3D para representar pixeles en una imagen.

## Conclusión

Los arreglo son estructuras de datos fundamentales en programación que proveen un acceso eficiente a elementos en memoria mediante índices numéricos. Tener un buen dominio de arreglos, matrices y sus usos es indispensable para cualquier programador.

---

{{< callout icon="sparkles" >}}
¡Gracias por llegar hasta acá! Espero que este recorrido por el universo de la programación haya sido tan apasionante para vos como lo fue para mí escribirlo.

Nos encantaría escuchar lo que pensás, así que no te quedes callado/a – dejá tus comentarios, sugerencias y todas esas ideas copadas que seguro se te ocurrieron.

Y para ir más allá de estas líneas, date una vuelta por los ejemplos prácticos que preparamos para vos. Vas a encontrar todo el código y los proyectos en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

¡Gracias por ser parte de esta comunidad de aprendizaje. Seguí programando y explorando nuevos territorios en este fascinante mundo del software!
{{< /callout >}}
