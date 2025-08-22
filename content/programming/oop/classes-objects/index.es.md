---
weight: 2
series: ["Programación: Aprendiendo Orientación a Objetos"]
series_order: 2
title: "Clases y Objetos"
authors:
  - jnonino
description: >
  En programación orientada a objetos, las clases y objetos son los conceptos centrales para entender cómo modelamos elementos de la realidad y definimos su estructura y comportamiento dentro del software. Veamos en detalle la anatomía de una clase, cómo crear objetos a partir de ella para usar sus propiedades y métodos, y otros detalles clave de su relación.
date: 2023-10-02
tags: ["Programación", "Programación Orientada a Objetos", "OOP", "Clases", "Objetos"]
---

## Anatomía de una clase

Una clase actúa como un **plano o molde** para construir objetos similares, definiendo sus características comunes y funcionalidades. Es similar al plano para construir casas de un mismo barrio: todas comparten ciertos atributos clave.

Los componentes típicos de una clase son:

**Atributos (propiedades)**: Variables que caracterizan al objeto. Por ejemplo, para una clase `Persona`, atributos como `nombre`, `edad`, `DNI`, etc.

```python
class Persona:
  dni = ""
  nombre = ""
  edad = 0
```

**Métodos**: Funciones que definen comportamientos. Por ejemplo, una `Persona` puede `caminar()`, `hablar()`, `comer()`, etc. Acceden a los atributos para implementar dicha funcionalidad.

**Constructor**: Método especial `__init__()` que se ejecuta al instanciar la clase y permite inicializar los atributos.

**Destructor**: Método `__del__()` que se ejecuta al eliminar la instancia liberando recursos. Opcional en algunos lenguajes.

---

## Creando objetos

A partir de la clase generamos **objetos**, que son **instancias concretas** con sus propios atributos definidos. Digamos que la clase Casa es el plano, y una casa específica en una calle determinada es el objeto.

En código creamos un objeto invocando la clase como si fuera un método:

```python
# Clase Persona
class Persona:
    def __init__(self, n, e):
        self.nombre = n
        self.edad = e

# Objeto Persona específico
pepe = Persona("Pepe", 30)
juan = Persona("Juan", 35)
```

Cada objeto comparte la estructura y comportamiento general, pero puede almacenar distintos datos.

### Utilizando Propiedades y Métodos

Ya tenemos una clase `Persona` y un objeto `pepe` de tipo `Persona`. ¿Cómo interactuamos con el objeto?

- **Propiedades**: Es posible acceder al valor de un atributo del objeto utilizando la referencia al objeto (`pepe`) y el nombre del atributo.

```python
pepe.nombre  # "Pepe"
pepe.edad    # 30
```

- **Métodos**: De la misma manera en la que se accede a los atributos pero agregando un paréntesis dentro del cual se pasan los argumentos si es que recibe alguno.

```python
# Clase Persona
class Persona:
    def __init__(self, n, e):
        self.nombre = n
        self.edad = e

    def comer(self, comida):
        print(f"Comiendo {comida}")

# Objeto Persona específico
pepe = Persona("Pepe", 30)
pepe.comer("pizza") # Imprime "Comiendo pizza"
```

El objeto pepe tiene ahora **estado** (propiedades) y **comportamiento** (métodos) propios.

---

## Self vs This

Un detalle importante en los métodos es cómo acceden a los atributos y otros métodos del objeto. Aquí entra otra diferencia entre lenguajes:

- **Self**: En Python, los atributos y métodos se acceden dentro de la clase anteponiendo `self`. Esto apunta al objeto instanciado.

```python
class Persona:

  def __init__(self, nombre):
    self.nombre = nombre

  def saludar(self):
    print(f"Hola! Soy {self.nombre}")

juan = Persona("Juan")
juan.saludar()
# Imprime "Hola! Soy Juan"
```

- **This**: En Java o C#, se utiliza `this` en lugar de self. Cumple la misma funcionalidad de apuntar a los miembros del objeto.

```java
public class Person {

  private String nombre;

  public Person(String nombre) {
    this.nombre= nombre;
  }

  public void saludar() {
    System.out.println("Hola! Soy " + this.nombre);
  }
}

Person juan = new Person("Juan");
juan.saludar();
// Imprime "Hola! Soy Juan"
```

---

## Conclusión

Las clases y objetos son los conceptos clave de la POO, permitiendo modelar entidades de la realidad y generar componentes modulares y genéricos de nuestro sistema para construir programas más robustos y fáciles de entender y mantener.

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
