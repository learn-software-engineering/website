---
weight: 5
series: ["Estructuras de Datos"]
series_order: 5
title: "Pilas"
authors:
  - jnonino
description: >
  Las pilas (stacks en inglés) son una estructura de datos abstracta que funciona bajo el principio LIFO (last in, first out), donde el último elemento en entrar es el primero en salir.
date: 2023-11-02
tags: ["Programación", "Estructuras de Datos", "Listas", "Listas Enlazadas", "Pilas"]
---

La naturaleza **LIFO** de las pilas se debe a que sólo se puede acceder y manipular el elemento superior. La operación de colocar un elemento sobre la pila se conoce como *"push"*, mientras que sacar un elemento de la pila es un *"pop"*. El funcionamiento LIFO provoca que el último elemento colocado en una pila sea el primero en ser retirado.

{{< figure
    src="stacks.jpg"
    alt="Diagrama de una pila"
    caption="Diagrama de una pila"
    >}}

---

## Operaciones principales

Las operaciones primarias que soporta una estructura de pila son:

- **Push:** agrega un elemento encima de la pila.
- **Pop:** saca el elemento de la pila que se encuentra en la cima.
- **Peek:** permite acceder al elemento de la cima sin sacarlo de la pila.
- **isEmpty:** consulta si la pila se encuentra vacía.

La mayoría de los lenguajes como Python y Java proveen implementaciones de pilas en sus librerías estándar.

---

## Implementación

Una pila puede implementarse utilizando una lista enlazada de manera que cada node apunte al nodo anterior.

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/main/programming/data_structures/stacks.py"
    type="python"
    >}}

---

## Ejemplos de uso

Las pilas tienen muchos usos en programación:

- **Pila de ejecución (call stack)**: registra las llamadas a funciones pendientes de resolver. Implementa el comportamiento LIFO esperado.

- **Pila de navegador**: permite volver atrás (undo) en el historial de navegación de forma similar a una pila LIFO.

- **Ejecución de expresiones matemáticas**: mediante una pila se puede verificar paréntesis, corchetes, llaves, etc.

- **Algoritmos y estructuras de datos**: como en el algoritmo quicksort y en la implementación de buses de datos (datapaths).

---

## Conclusión

Las pilas son estructuras de datos versátiles gracias a su principio de funcionamiento LIFO. Tener un buen dominio de pilas, sus usos y aplicaciones es esencial en la ciencia de la computación.

---

{{< alert icon="comment" cardColor="white" iconColor="black" textColor="black" >}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}
