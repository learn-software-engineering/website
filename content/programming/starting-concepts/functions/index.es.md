---
weight: 4
series: ["Programación: Aprendiendo los Conceptos Iniciales"]
series_order: 4
title: "Funciones"
authors:
  - jnonino
description: >
  En el vasto y emocionante mundo de la programación, existen conceptos que son pilares fundamentales para cualquier desarrollador, sin importar su nivel de experiencia. Uno de estos conceptos es el de las funciones. ¿Qué son? ¿Por qué son tan cruciales? ¡Vamos a descubrirlo!
date: 2023-09-29
tags: ["Programación", "Conceptos Iniciales", "Funciones", "Métodos"]
---

## ¿Qué son las funciones?

Una función, en términos simples, es un bloque de código que se ejecuta sólo cuando es llamado. Puedes pensar en ella como un pequeño programa dentro de tu programa principal, diseñado para realizar una tarea específica[^1]. Una función también puede verse como una caja negra: le pasamos una entrada (parámetros), ocurre algún procesamiento interno, y produce una salida (retorno).

Las funciones nos permiten segmentar nuestro código en partes lógicas, donde cada parte realiza una única acción. Esto brinda varios beneficios[^2]:

- **Reutilización**: Una vez definida la función, podemos ejecutar (llamar) ese código desde cualquier lugar de nuestro programa cuantas veces sea necesario.
- **Organización**: Permite dividir un programa grande en partes más pequeñas y manejables.
- **Encapsulamiento**: Las funciones reducen la complejidad escondiendo los detalles de implementación internos.
- **Mantenimiento**: Si necesitamos realizar cambios, solo debemos modificar el código en un lugar (la función) en lugar de rastrear todas las instancias de ese código.

**Procedimientos vs. Funciones**

Es vital distinguir entre estos dos conceptos. Mientras que una función siempre devuelve un valor, un procedimiento realiza una tarea pero no devuelve nada. En algunos lenguajes, esta diferencia es más clara que en otros. Python, por ejemplo, tiene funciones que pueden o no devolver valores.

---

## Anatomía de una función

En Python, una función se declara usando la palabra clave `def`, seguida del nombre de la función y paréntesis. El código dentro de la función se denomina el *cuerpo* de la función[^4] y contiene el conjunto de instrucciones a ejecutar para cumplir con su tarea..

```python
def mi_funcion():
    print("¡Hola desde mi función!")
```

Para **llamar** o **invocar** una función, simplemente usamos su nombre seguido de paréntesis:

```python
mi_funcion()  # Salida: ¡Hola desde mi función!
```

### Parámetros y argumentos

Las funciones se vuelven aún más poderosas cuando les pasamos información, conocida como **parámetros**. Estos actúan como "*variables*" dentro de la función y permiten que la función trabaje con diferentes datos cada vez que se llama.

Mientras que los **parámetros** son variables definidas en la definición de la función. Los **argumentos** son los valores reales pasados al llamar a la función.

```python
def saludo(nombre):
    print(f"¡Hola {nombre}!")


saludo("María")
# Salida:
#   ¡Hola María!
```
Podemos definir valores por defecto para los parámetros
Python permite **parámetros por defecto**, que tienen un valor predeterminado, lo cual hace opcional pasar esos argumentos al llamar la función. También permite **parámetros nombrados** que permiten pasar los argumentos en cualquier orden, especificando su nombre.

```python
def saludo(nombre="María", repeticiones=3):
    repeticion = 1
    while repeticion <= repeticiones:
        print(f"¡Hola {nombre}!")
        repeticion += 1


saludo()
# Salida:
#   ¡Hola María!
#   ¡Hola María!
#   ¡Hola María!


saludo("Florencia", 4)
# Salida:
#   ¡Hola Florencia!
#   ¡Hola Florencia!
#   ¡Hola Florencia!
#   ¡Hola Florencia!


saludo(repeticiones=2, nombre="Julián")
# Salida
#   ¡Hola Julián!
#   ¡Hola Julián!
```

### Retorno de valores

Las funciones pueden devolver un resultado o valor de retorno usando la palabra reservada `return`.

```python
def area_circulo(radio):
    return 3.14 * (radio ** 2)


resultado = area_circulo(10)
print(resultado) # Salida: 314
```

El valor de retorno se pasa de vuelta a donde se llamó la función y se puede asignar a una variable para usarlo.

Las funciones también pueden ejecutar alguna tarea sin devolver nada explícitamente. En Python esto se conoce como retornar `None`.

---

## Variables locales y globales

Las **variables locales** se definen dentro de una función y solo existen en ese ámbito, mientras que las **variables globales** están definidas fuera y pueden ser accedidas desde cualquier parte del código. Es crucial entender su **alcance** (dónde puede ser accesible una variable) y **duración** (cuánto tiempo vive una variable).

```python
x = 10 # x es global

def suma():
    y = 5 # y es local
    return x + y

suma()    # Salida: 15
print(y)  # Error, y no existe fuera de la función
```

Podemos leer variables globales desde una función, pero si necesitamos modificarla debemos declararla `global`.

```python
x = 10

def suma():
    global x
    x = x + 5

suma()
print(x) # 15
```

{{% alert color="warning" %}}
Idealmente las funciones solo deben trabajar con variables locales y parámetros. Limita el uso de variables globales. Si bien pueden ser útiles, también pueden hacer que el código sea difícil de leer y mantener.
{{% /alert %}}

---

## Buenas prácticas

Al crear funciones debemos seguir ciertos principios y patrones[^3]:

- El nombre de una función debe indicar claramente su propósito.
- Hacer las funciones pequeñas, simples y enfocadas en una tarea. Una función debe hacer una cosa y hacerla bien.
- Utilizar nombres descriptivos para las funciones y sus parámetros.
- Evitar efectos secundarios y modificación de variables globales.
- Documentar adecuadamente el propósito y uso de cada función.
- Limitar el número de parámetros, idealmente de 0 a 3 parámetros.

Seguir estas buenas prácticas nos ayudará a crear funciones reutilizables, encapsuladas y fáciles de mantener.

---

## Conclusión

Las funciones son componentes fundamentales en la programación, permitiéndonos organizar, reutilizar y encapsular código. Definiendo funciones que realicen una sola tarea mantenemos nuestros programas simplificados, fáciles de entender y modificar. Al comprender y dominar este concepto, no solo mejoras la calidad de tu código sino también tu eficiencia como desarrollador.

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

---

## Referencias

[^1]: McConnell, S. (2004). *Code Complete*. Microsoft Press.
[^2]: Joyanes Aguilar, L. (2008). *Fundamentos de programación: algoritmos, estructura de datos y objetos*. McGraw-Hill.
[^3]: Kindler, E., & Krivy, I. (2011). *Object-Oriented Simulation of systems with Java: A working introduction*. BoD–Books on Demand.
[^4]: Python Software Foundation. (2022). *Documentación oficial de Python*.
