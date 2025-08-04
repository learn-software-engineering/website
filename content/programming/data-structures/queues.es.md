---
weight: 6
series: ["Estructuras de Datos"]
series_order: 6
title: "Colas"
authors:
  - jnonino
description: >
  Las colas (queues en inglés) son una estructura de datos abstracta que funciona bajo el principio FIFO (first in, first out), donde el primer elemento en entrar es también el primero en salir. Las colas se utilizan para ordenar elementos de forma que el que llega primero es procesado primero. Comprender su funcionamiento es esencial para cualquier programador.
date: 2023-11-03
tags: ["Programación", "Estructuras de Datos", "Listas", "Listas Enlazadas", "Colas"]
---

La naturaleza FIFO (first in, first out) de las colas se debe a que sólo se puede acceder y manipular el elemento inicial. Cuando se agrega un elemento a la cola se conoce como *"enqueue"*, mientras que eliminar un elemento se denomina *"dequeue"*.

Esto hace que el primer elemento en ser añadido a la cola también sea el primero en ser retirado, de ahí su comportamiento FIFO.

![Diagrama de una cola](/images/content/programming/0400-data-structures-1/diagram-queues.jpg)

---

## Operaciones principales

Las operaciones básicas de una cola son:

- **Enqueue:** Agrega un elemento al final de la cola.
- **Dequeue:** Saca el elemento del frente de la cola.
- **Peek:** Obtiene el elemento al frente sin sacarlo.
- **isEmpty:** Consulta si la cola está vacía.

---

## Implementación

Al igual que las pilas, las colas se pueden implementar usando listas enlazadas.
Se agrega al final y se saca del frente manteniendo referencias a ambos extremos.

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class Queue:
    def __init__(self):
        self.front = None
        self.end = None
        self.size = 0

    def enqueue(self, value):
        new_node = Node(value)
        if self.end is None:
            self.end = new_node
            self.front = new_node
            return
        self.end.next = new_node
        self.end = new_node
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            return None
        value = self.front.value
        self.front = self.front.next
        if self.front is None:
            self.end = None
        self.size -= 1
        return value

    def peek(self):
        if self.is_empty():
            return None
        return self.front.value

    def is_empty(self):
        return self.front is None  # Returns true if front is None

    def __len__(self):
        return self.size

    def __str__(self):
        values = []
        current = self.front
        while current:
            values.append(str(current.value))
            current = current.next
        return "\n".join(values)
```
---

## Ejemplos de uso

Algunos usos comunes de colas:

- Colas de impresión donde primero en entrar, primero en imprimir.
- Colas de tareas en sistemas operativos para orden de ejecución.
- Simulaciones donde se debe respetar orden de llegada como en bancos.
- Canales de mensajes como los de RabbitMQ o Kafka.
- Buffers circulares en audio para streaming.

---

## Conclusión

Las colas son estructuras versátiles gracias a su principio FIFO. Tener un buen manejo de colas, implementación y aplicaciones reforzará tus habilidades como programador.

---

{{% alert color="info" %}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{% /alert %}}
