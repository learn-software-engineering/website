---
weight: 2
series: ["Programación: Aprendiendo las Bases"]
series_order: 2
title: "Computadoras"
authors:
  - jnonino
description: >
  En el mundo actual, donde las tecnologías digitales se han convertido en una extensión de nuestras vidas diarias, resulta esencial comprender los cimientos sobre los que se construyen nuestras herramientas digitales. En esta ocasión, nos sumergiremos en el corazón de las computadoras para entender cómo funcionan y por qué son esenciales en nuestra sociedad moderna.
date: 2023-09-02
tags: ["Programación", "Computadoras", "Hardware", "Software", "CPU", "Memoria"]
---

Si alguna vez te has preguntado qué es lo que hace "tictac" dentro de esa caja metálica que llamamos computadora, estás en el lugar indicado. En esencia, una computadora es una combinación de hardware y software que trabajan juntos para llevar a cabo tareas específicas.

El ***hardware*** es toda la parte física de la computadora: el CPU (que es como el cerebro de la máquina), la RAM (donde la computadora guarda la información con la que está trabajando en un momento determinado), dispositivos de almacenamiento (donde se guardan los datos de forma permanente) y periféricos (como el teclado, el mouse o la pantalla)[^1].

Por otro lado, el ***software*** es el conjunto de instrucciones que le dice al hardware qué hacer. Hay varios tipos de software, desde el software del sistema, como el sistema operativo que coordina todas las acciones de la máquina, hasta software de aplicación que nos permite hacer cosas como escribir documentos o jugar videojuegos[^2].

El papel del sistema operativo es crucial. Es el mediador entre el usuario y el hardware, asegurando que todo funcione armónicamente. Si el hardware fuera una orquesta, el sistema operativo sería el director que asegura que cada instrumento toque en el momento y de la manera correcta.

---

## El sistema binario: el lenguaje secreto de las computadoras

A diferencia de nosotros, que usamos un sistema decimal basado en diez dígitos (del 0 al 9), las computadoras usan el sistema binario, que solo tiene dos dígitos: 0 y 1. ¿Por qué? Bueno, en el nivel más básico, una computadora está hecha de millones de transistores que pueden estar en uno de dos estados: encendido o apagado. Estos estados se representan con esos dígitos: 0 para apagado y 1 para encendido[^3].

Los términos "bit" y "byte" son fundamentales aquí. Un *bit* es la unidad más pequeña de datos en una computadora y puede tener un valor de 0 o 1. Un *byte*, por otro lado, es un conjunto de 8 bits y puede representar 256 valores diferentes (desde \\(00000000\\) hasta \\(11111111\\) en binario)[^4].

Este sistema binario no solo representa números, sino también texto, imágenes y cualquier tipo de dato. Por ejemplo, en el código ASCII (un estándar de codificación de caracteres), la letra "A" se representa como \\(01000001\\) en binario.

En un artículo posterior, vamos a hablar en detalle acerca del sistema binario y otro sistema muy utilizado en el ambiente de la computación, el *hexadecimal*.

---

## Memoria y almacenamiento: donde residen nuestros datos

La memoria y el almacenamiento son dos conceptos cruciales en la informática. Si bien a menudo se usan indistintamente, tienen roles muy diferentes.

La memoria, específicamente la RAM, es volátil. Esto significa que la información se pierde cuando apagamos la computadora. La RAM es esencialmente el "*espacio de trabajo*" de la computadora, donde almacena datos e instrucciones mientras está en uso. Hay varios tipos de RAM, siendo DRAM y SRAM los más comunes[^5].

Por otro lado, tenemos la ROM (Memoria de Solo Lectura). A diferencia de la RAM, la ROM no es volátil y se utiliza para almacenar firmware, es decir, software que está íntimamente ligado al hardware y que no necesita cambios frecuentes.

En cuanto al almacenamiento, dispositivos como discos duros, SSDs y unidades flash nos permiten guardar información de manera permanente. Estos dispositivos forman parte de lo que se conoce como la jerarquía de memoria, que va desde la memoria caché (rápida pero pequeña) hasta el almacenamiento secundario (más lento pero con gran capacidad)[^6].

---

{{< alert icon="comment" cardColor="white" iconColor="black" textColor="black" >}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}

---

## Referencias

[^1]: Patterson, D. & Hennessy, J. (2014). *Arquitectura de Computadoras*. Buenos Aires: Ediciones Omega.
[^2]: Silberschatz, A., Galvin, P. B., & Gagne, G. (2009). *Fundamentos de Sistemas Operativos*. Buenos Aires: Ediciones M.
[^3]: Tanenbaum, A. (2012). *Estructura de Computadoras*. Buenos Aires: Prentice Hall.
[^4]: Brookshear, J. G. (2011). *Ciencia de la Computación: Una visión general*. Buenos Aires: Pearson Educación.
[^5]: Jacob, B., Ng, S. W., & Wang, D. T. (2007). *Arquitecturas de memoria en sistemas de computadoras*. Buenos Aires: Wiley-Interscience.
[^6]: Siewiorek, D. P. & Swarz, R. S. (2017). *Principios de diseño de sistemas computacionales*. Buenos Aires: Morgan Kaufmann.
