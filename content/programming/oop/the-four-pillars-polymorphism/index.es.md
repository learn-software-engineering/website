---
weight: 14
series: ["Programación: Aprendiendo Orientación a Objetos"]
series_order: 14
title: "Los Cuatro Pilares: Polimorfismo"
authors:
  - jnonino
description: >
  El polimorfismo es un concepto fundamental en la programación orientada a objetos que permite que objetos de diferentes clases sean tratados como objetos de una clase base común. El término "polimorfismo" proviene del griego y significa "muchas formas". En POO, se refiere a la capacidad de una única interfaz para representar diferentes formas subyacentes (tipos de datos o clases).
date: 2024-10-12
tags: ["Programación", "Programación Orientada a Objetos", "OOP", "Los Cuatro Pilares", "Polimorfismo"]
---

El polimorfismo permite escribir código flexible y reutilizable al permitirnos trabajar con objetos a un nivel más abstracto, sin necesidad de conocer sus tipos específicos.

Existen dos tipos principales de polimorfismo en la programación orientada a objetos:

1. **Polimorfismo en tiempo de compilación (Polimorfismo estático)**
   - Se logra a través de la *sobrecarga* de métodos.
   - Se resuelve en tiempo de compilación.

2. **Polimorfismo en tiempo de ejecución (Polimorfismo dinámico)**
   - Se logra a través de la *sobrescritura* de métodos.
   - Se resuelve en tiempo de ejecución.

Python admite principalmente el polimorfismo en tiempo de ejecución, ya que es un lenguaje de tipado dinámico. Sin embargo, podemos demostrar conceptos similares al polimorfismo en tiempo de compilación también.

Exploremos diferentes aspectos del polimorfismo en Python:

### Duck typing

Python utiliza el duck typing, que es una forma de polimorfismo. La idea es: "Si camina como un pato y grazna como un pato, entonces debe ser un pato". En otras palabras, Python se preocupa más por los métodos que tiene un objeto que por el tipo del objeto en sí.

```python
class Pato:
    def hablar(self):
        return "¡Cuac cuac!"

class Perro:
    def hablar(self):
        return "¡Guau guau!"

class Gato:
    def hablar(self):
        return "¡Miau miau!"

def sonido_animal(animal):
    return animal.hablar()

# Uso
pato = Pato()
perro = Perro()
gato = Gato()

print(sonido_animal(pato))   # Salida: ¡Cuac cuac!
print(sonido_animal(perro))  # Salida: ¡Guau guau!
print(sonido_animal(gato))   # Salida: ¡Miau miau!
```

En este ejemplo, `sonido_animal()` funciona con cualquier objeto que tenga un método `hablar()`, independientemente de su clase.

### Sobrescritura de métodos

La sobrescritura de métodos es un aspecto clave del polimorfismo en tiempo de ejecución. Ocurre cuando una clase derivada define un método con el mismo nombre que un método en su clase base.

```python
class Figura:
    def area(self):
        pass

class Rectangulo(Figura):
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto

    def area(self):
        return self.ancho * self.alto

class Circulo(Figura):
    def __init__(self, radio):
        self.radio = radio

    def area(self):
        return 3.14159 * self.radio ** 2

# Uso
figuras = [Rectangulo(5, 4), Circulo(3)]

for figura in figuras:
    print(f"Área: {figura.area()}")

# Salida:
# Área: 20
# Área: 28.27431
```

Aquí, `Rectangulo` y `Circulo` sobrescriben el método `area()` de la clase `Figura`.

### Sobrecarga de operadores

Python permite la sobrecarga de operadores, que es una forma de polimorfismo en tiempo de compilación. Permite que el mismo operador tenga diferentes significados según los operandos.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, otro):
        return Vector(self.x + otro.x, self.y + otro.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# Uso
v1 = Vector(2, 3)
v2 = Vector(3, 4)
v3 = v1 + v2

print(v3)  # Salida: Vector(5, 7)
```

Aquí, hemos sobrecargado el operador `+` para nuestra clase `Vector`.

### Clases base abstractas

El módulo `abc` de Python proporciona infraestructura para definir clases base abstractas, que son una forma poderosa de definir interfaces en Python.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def hacer_sonido(self):
        pass

class Perro(Animal):
    def hacer_sonido(self):
        return "¡Guau!"

class Gato(Animal):
    def hacer_sonido(self):
        return "¡Miau!"

# Uso
def sonido_animal(animal):
    return animal.hacer_sonido()

perro = Perro()
gato = Gato()

print(sonido_animal(perro))  # Salida: ¡Guau!
print(sonido_animal(gato))   # Salida: ¡Miau!

# Esto generará un TypeError
# animal = Animal()
```

Las clases base abstractas no pueden ser instanciadas y obligan a las clases derivadas a implementar ciertos métodos, asegurando una interfaz consistente.

### Aplicaciones en el mundo real

El polimorfismo se utiliza ampliamente en aplicaciones del mundo real:

1. **Frameworks de GUI**: Diferentes widgets (botones, cajas de texto) pueden responder a eventos comunes (clic, hover) de sus propias maneras.
2. **Interfaces de bases de datos**: Diferentes sistemas de bases de datos pueden implementar una interfaz común para consultas, permitiendo que las aplicaciones trabajen con varias bases de datos sin cambiar el código.
3. **Sistemas de plugins**: Las aplicaciones pueden trabajar con plugins a través de una interfaz común, independientemente de la implementación específica de cada plugin.
4. **Desarrollo de juegos**: Diferentes entidades del juego pueden compartir comportamientos comunes (mover, colisionar) pero implementarlos de manera diferente.

Aquí hay un ejemplo simple de un sistema de plugins:

```python
class Plugin(ABC):
    @abstractmethod
    def procesar(self, datos):
        pass

class PluginMayusculas(Plugin):
    def procesar(self, datos):
        return datos.upper()

class PluginInvertir(Plugin):
    def procesar(self, datos):
        return datos[::-1]

class Aplicacion:
    def __init__(self):
        self.plugins = []

    def agregar_plugin(self, plugin):
        self.plugins.append(plugin)

    def procesar_datos(self, datos):
        for plugin in self.plugins:
            datos = plugin.procesar(datos)
        return datos

# Uso
app = Aplicacion()
app.agregar_plugin(PluginMayusculas())
app.agregar_plugin(PluginInvertir())

resultado = app.procesar_datos("Hola, Mundo!")
print(resultado)  # Salida: !ODNUM ,ALOH
```

Este ejemplo demuestra cómo el polimorfismo permite que la clase `Aplicacion` trabaje con diferentes plugins a través de una interfaz común.

---

## Referencias

1. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
2. Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design. Prentice Hall.
3. Phillips, D. (2010). Python 3 Object Oriented Programming. Packt Publishing.
4. Lutz, M. (2013). Learning Python: Powerful Object-Oriented Programming. O'Reilly Media.
5. Ramalho, L. (2015). Fluent Python: Clear, Concise, and Effective Programming. O'Reilly Media.
6. Van Rossum, G., Warsaw, B., & Coghlan, N. (2001). PEP 8 -- Style Guide for Python Code. Python.org. https://www.python.org/dev/peps/pep-0008/
7. Python Software Foundation. (n.d.). The Python Standard Library. Python.org. https://docs.python.org/3/library/

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
