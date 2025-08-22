---
weight: 6
series: ["Programación: Aprendiendo Estructuras de Datos"]
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

{{< figure
    src="images/queues.jpg"
    alt="Diagrama de una cola"
    caption="Diagrama de una cola"
    >}}

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

{{< codeimporter
    url="https://raw.githubusercontent.com/learn-software-engineering/examples/main/programming/data_structures/queues.py"
    type="python"
    >}}

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

{{< alert icon="comment" >}}
¡Gracias por haber llegado hasta acá!

Si te gustó el artículo, por favor ¡no olvides compartirlo con tu familia, amigos y colegas!

Y si puedes, envía tus comentarios, sugerencias, críticas a nuestro mail o por redes sociales, nos ayudarías a generar mejor contenido y sobretodo más relevante para vos.

[{{< icon "email" >}}](mailto:learn.software.eng@gmail.com)
[{{< icon "github" >}}](https://github.com/learn-software-engineering)
[{{< icon "patreon" >}}](https://patreon.com/learnsoftwareeng)
[{{< icon "linkedin" >}}](https://linkedin.com/company/learn-software)
[{{< icon "instagram" >}}](https://www.instagram.com/learnsoftwareeng)
[{{< icon "facebook" >}}](https://www.facebook.com/learn.software.eng)
[{{< icon "x-twitter" >}}](https://x.com/software45687)
{{< /alert >}}
