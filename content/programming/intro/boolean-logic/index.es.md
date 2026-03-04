---
weight: 4
title: "Lógica Booleana"
cardImage: "featured.es.jpg"
authors:
  - jnonino
description: >
  En la vida, a menudo buscamos certezas. ¿Es cierto o falso que va a llover mañana? ¿Es correcto o incorrecto actuar de cierta manera? Esta dicotomía, esta división entre dos estados opuestos, se encuentra en el núcleo mismo de una rama esencial de las matemáticas y la informática: la lógica booleana.
date: 2023-09-09
tags: ["Programación", "Lógica Booleana"]
---

La lógica booleana, nombrada en honor a [George Boole](https://es.wikipedia.org/wiki/George_Boole), un matemático inglés del siglo XIX, es un sistema matemático que se ocupa de operaciones que tienen solo dos resultados posibles: verdadero o falso, representados generalmente como 1 y 0, respectivamente. En su obra "An Investigation of the Laws of Thought", Boole estableció las bases de esta lógica, presentando un sistema algebraico que podría utilizarse para representar estructuras lógicas.

## Operaciones Booleanas

Dentro de la lógica booleana, existen operaciones fundamentales que permiten manipular y combinar estas expresiones binarias:

1. **AND (Y)**: Esta operación devuelve verdadero (1) solo si ambas entradas son verdaderas. Por ejemplo, si tenemos dos interruptores, ambos deben estar en la posición *encendido* para que una luz se encienda.

2. **OR (O)**: Devuelve verdadero si al menos una de las entradas es verdadera. Siguiendo con el ejemplo de los interruptores, con que uno de ellos esté *encendido*, la luz se iluminará.

3. **NOT (NO)**: Es una operación unaria, lo que significa que solo tiene una entrada. Simplemente invierte el valor de entrada. Si le das un 1, devuelve un 0 y viceversa.

4. **NAND (NO Y)**: Es la negación de AND. Solo devuelve falso si ambas entradas son verdaderas.

5. **NOR (NO O)**: Es la negación de OR. Devuelve verdadero solo si ambas entradas son falsas.

6. **XOR (O exclusivo)**: Devuelve verdadero si las entradas son diferentes. Si ambas son iguales, devuelve falso.

7. **XNOR (NO O exclusivo)**: Es la negación de XOR. Devuelve verdadero si ambas entradas son iguales.

## La importancia de esta lógica en computación y programación

La computación moderna, en su esencia, es la manipulación de bits, esos unos y ceros que mencionamos. Cada operación que realiza una computadora, desde simples cálculos hasta la renderización de gráficos complejos, implica operaciones booleanas en algún nivel.

En programación, la lógica booleana se utiliza en estructuras de control, como condiciones (if, else) y bucles, permitiendo a los programas tomar decisiones basadas en ciertas condiciones.

## Tablas de verdad: el mapa de la lógica Booleana

Una tabla de verdad es una representación gráfica de una operación booleana. Enumera todas las combinaciones posibles de entradas y muestra el resultado de la operación para cada combinación.

Por ejemplo,

| A | B | A AND B | A OR B | A XOR B | A NOR B | A NAND B | NOT A | A NXOR B |
|:-:|:-:|:-------:|:------:|:-------:|:-------:|:--------:|:-----:|:--------:|
| 1 | 1 | 1       | 1      | 0       | 0       | 0        | 0     | 1        |
| 1 | 0 | 0       | 1      | 1       | 0       | 1        | 0     | 0        |
| 0 | 1 | 0       | 1      | 1       | 0       | 1        | 1     | 0        |
| 0 | 0 | 0       | 0      | 0       | 1       | 1        | 1     | 1        |

## Conclusiones

La lógica booleana es mucho más que un conjunto de reglas matemáticas abstractas. Es el lenguaje fundamental de las máquinas, el código que subyace a la era digital en la que vivimos. Al comprender sus principios, no solo nos volvemos más adeptos a trabajar con tecnología, sino que también adquirimos una apreciación más profunda de las estructuras que sustentan nuestro mundo digital.

---

{{< callout icon="sparkles" >}}
¡Gracias por llegar hasta acá! Espero que este recorrido por el universo de la programación haya sido tan apasionante para vos como lo fue para mí escribirlo.

Nos encantaría escuchar lo que pensás, así que no te quedes callado/a – dejá tus comentarios, sugerencias y todas esas ideas copadas que seguro se te ocurrieron.

Y para ir más allá de estas líneas, date una vuelta por los ejemplos prácticos que preparamos para vos. Vas a encontrar todo el código y los proyectos en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

¡Gracias por ser parte de esta comunidad de aprendizaje. Seguí programando y explorando nuevos territorios en este fascinante mundo del software!
{{< /callout >}}
