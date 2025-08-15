---
weight: 3
series: ["Introducción a la Programación"]
series_order: 3
title: "Sistemas de Numeración"
authors:
  - jnonino
description: >
  En nuestro día a día, estamos rodeados de números. Desde la hora que marca el reloj cuando suena la alarma por la mañana, hasta el precio del café que compramos en la esquina. Pero, ¿alguna vez te detuviste a pensar en la naturaleza de esos números? En este artículo, nos sumergiremos en el fascinante mundo de los sistemas de numeración y aprenderemos cómo, dependiendo del contexto, un número puede tener múltiples representaciones.
date: 2023-09-05
tags: ["Programación", "Sistemas de Numeración", "Decimal", "Binario", "Hexadecimal"]
---

## El sistema decimal: la base de nuestra cotidianidad

Desde pequeños, nos enseñan a contar usando diez dígitos: del 0 al 9. Este sistema, conocido como decimal, es la base de casi todas nuestras actividades matemáticas y financieras, desde sumar cuentas hasta calcular intereses en el banco[^1]. Tiene su origen en la cantidad de dedos que tenemos en las manos, lo que lo convierte en el sistema más intuitivo y natural para nosotros. Pero lo que lo hace especial es su naturaleza posicional.

Para comprender este concepto, consideremos el número 237:

- El 7, situado a la derecha, está en la posición de las *unidades*. Es decir, \\(7 \times 10^0\\) (cualquier número elevado a la potencia de 0 es 1). Por lo tanto, su valor es simplemente 7.
- El 3, en la posición del medio, representa las *decenas*, es decir, \\(3 \times 10^1 = 3 \times 10 = 30\\).
- El 2, el número más a la izquierda, está en la posición de las *centenas*, traduciéndose a \\(2 \times 10^2 = 2 \times 100 = 200\\).

Si sumamos estos valores,

```math
2 \times 10^2 + 3 \times 10^1 + 7 \times 10^0 = 200 + 30 + 7 = 237
```

---

## El sistema binario: el lenguaje secreto de las computadoras

Si bien el sistema decimal domina nuestra vida cotidiana, las máquinas que usamos todos los días, desde nuestros celulares hasta las computadoras, operan en un mundo completamente diferente: el mundo binario. En este sistema, solo existen dos dígitos: 0 y 1. A primera vista, puede parecer limitante, pero este sistema es la esencia de la electrónica digital. Los dispositivos electrónicos, con sus millones de transistores, operan usando estos dos estados: encendido (1) y apagado (0)[^2].

A pesar de su aparente simplicidad, el sistema binario puede representar cualquier cantidad o información que el sistema decimal pueda expresar. Por ejemplo, el número decimal 5 se representa como 101 en binario.

El sistema binario, con sus unos y ceros, opera de manera similar al sistema decimal, pero en lugar de potencias de 10, usa potencias de 2.

Tomemos el número binario 1011:

- El bit más a la derecha representa \\(1 \times 2^0 = 1\\)
- El siguiente bit representa \\(1 \times 2^1 = 2\\)
- Luego viene \\(0 \times 2^2 = 0\\)
- El bit más a la izquierda en este número representa \\(1 \times 2^3 = 8\\)

Entonces, 1011 en binario se traduce a decimal de la siguiente manera:

```math
1011 = 1 \times 2^3 + 0 \times 2^2 + 1 \times 2^1 + 1 \times 2^0 = 8 + 0 + 2 + 1 = 11
```

---

## El sistema hexadecimal: un puente entre humanos y máquinas

Mientras que el sistema binario es perfecto para las máquinas, puede ser un poco engorroso para nosotros, especialmente cuando tratamos con números binarios largos. Aquí es donde entra el sistema hexadecimal, que utiliza dieciséis dígitos distintos: 0-9 y A-F, donde A representa 10, B es 11, y así sucesivamente hasta F que es 15[^3].

El hexadecimal es especialmente útil porque proporciona una forma más compacta de representar números binarios. Cada dígito hexadecimal corresponde a exactamente cuatro dígitos binarios (bits). Por ejemplo pensemos en la representación en binario del número 41279 y como el sistema hexadecimal consigue una representación más compacta:

```math
41279 = 1010 0001 0011 1111 = A13F
```

Pero el sistema hexadecimal es más que una representación compacta de números binarios, es un sistema de numeración posicional como el decimal o binario con base 16 en lugar de 10 o 2. Veamos como conseguir la representación decimal del número del ejemplo anterior (A13F).

- El dígito más a la derecha representa \\(F \times 16^0 = 15 \times 16^0 = 15\\)
- El siguiente representa \\(3 \times 16^1 = 48\\)
- Luego viene \\(1 \times 16^2 = 256\\)
- El dígito más a la izquierda en este número representa \\(A \times 16^3 = 10 \times 16^3 = 40960\\)

Entonces, A13F en hexadecimal se traduce a decimal de la siguiente manera:

```math
A13F = A \times 16^3 + 1 \times 16^2 + 3 \times 16^1 + F \times 16^0 = 10 \times 4096 + 1 \times 256 + 3 \times 16 + 15 \times 1 = 40960 + 256 + 48 + 15 = 41279
```

---

## Conclusión

Los sistemas de numeración son como lentes a través de los cuales vemos y entendemos el mundo de las matemáticas y la computación. Aunque en nuestra vida diaria el sistema decimal sea el rey, es esencial apreciar y comprender los sistemas binario y hexadecimal, especialmente en esta era digital.

Así que, la próxima vez que estés frente a tu computadora o usando una app en tu celular, recordá que detrás de esa interfaz amigable, hay un mundo binario en pleno funcionamiento, y que el sistema hexadecimal actúa como un traductor entre ese mundo y nosotros.

---

{{< alert icon="comment" cardColor="white" iconColor="black" textColor="black" >}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}

---

## Referencias

[^1]: Ifrah, G. (2000). Historia universal de las cifras. Buenos Aires: Editorial Taurus.
[^2]: Tanenbaum, A. (2012). Estructura de Computadoras. Buenos Aires: Prentice Hall.
[^3]: Knuth, D. (2007). El Arte de Programar Computadoras: Sistemas de numeración. Buenos Aires: Ediciones Omega.
