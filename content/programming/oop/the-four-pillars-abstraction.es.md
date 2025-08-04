---
weight: 15
series: ["Programación Orientada a Objetos"]
series_order: 15
title: "Los Cuatro Pilares: Abstracción"
authors:
  - jnonino
description: >
  La abstracción es el proceso de ocultar los detalles complejos de implementación y mostrar solo las características necesarias de un objeto. Se trata de crear una vista simplificada de un objeto que represente sus características esenciales sin incluir detalles de fondo o explicaciones.
date: 2024-10-12
tags: ["Programación", "Programación Orientada a Objetos", "OOP", "Los Cuatro Pilares", "Abstracción"]
---

Los aspectos clave de la abstracción incluyen:

1. **Simplificación**: La abstracción reduce la complejidad ocultando detalles innecesarios.
2. **Enfoque en características esenciales**: Enfatiza lo que hace un objeto en lugar de cómo lo hace.
3. **Separación de preocupaciones**: Permite separar la interfaz de una clase de su implementación.
4. **Modularidad**: La abstracción promueve el diseño modular al definir límites claros entre componentes.

### Clases abstractas e interfaces

En muchos lenguajes orientados a objetos, la abstracción se implementa a través de clases abstractas e interfaces. Aunque Python no tiene un concepto integrado de interfaz, podemos lograr una funcionalidad similar usando clases base abstractas. El módulo `abc` de Python proporciona infraestructura para definir clases base abstractas:

```python
from abc import ABC, abstractmethod

class Figura(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimetro(self):
        pass

class Rectangulo(Figura):
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto

    def area(self):
        return self.ancho * self.alto

    def perimetro(self):
        return 2 * (self.ancho + self.alto)

class Circulo(Figura):
    def __init__(self, radio):
        self.radio = radio

    def area(self):
        return 3.14159 * self.radio ** 2

    def perimetro(self):
        return 2 * 3.14159 * self.radio

# Uso
# figuras = [Figura()]  # Esto generaría TypeError
figuras = [Rectangulo(5, 4), Circulo(3)]

for figura in figuras:
    print(f"Área: {figura.area()}, Perímetro: {figura.perimetro()}")

# Salida:
# Área: 20, Perímetro: 18
# Área: 28.27431, Perímetro: 18.84954
```

En este ejemplo:
- `Figura` es una clase base abstracta que define la interfaz para todas las figuras.
- `Rectangulo` y `Circulo` son clases concretas que implementan la interfaz `Figura`.
- No podemos instanciar `Figura` directamente, pero podemos usarla como un tipo común para todas las figuras.

### Implementando abstracción en Python

Aunque Python proporciona clases base abstractas para definir interfaces formalmente, también podemos lograr la abstracción mediante convenciones y documentación. Veamos un ejemplo sin utilizar `ABC`:

```python
class BaseDeDatos:
    def conectar(self):
        raise NotImplementedError("La subclase debe implementar este método abstracto")

    def ejecutar(self, consulta):
        raise NotImplementedError("La subclase debe implementar este método abstracto")

class BaseDeDatosMySQL(BaseDeDatos):
    def conectar(self):
        print("Conectando a la base de datos MySQL...")

    def ejecutar(self, consulta):
        print(f"Ejecutando consulta MySQL: {consulta}")

class BaseDeDatosPostgreSQL(BaseDeDatos):
    def conectar(self):
        print("Conectando a la base de datos PostgreSQL...")

    def ejecutar(self, consulta):
        print(f"Ejecutando consulta PostgreSQL: {consulta}")

def realizar_operacion_en_bd(base_de_datos):
    base_de_datos.conectar()
    base_de_datos.ejecutar("SELECT * FROM usuarios")

# Uso
bd_mysql = BaseDeDatosMySQL()
bd_postgres = BaseDeDatosPostgreSQL()

realizar_operacion_en_bd(bd_mysql)
realizar_operacion_en_bd(bd_postgres)

# Salida:
# Conectando a la base de datos MySQL...
# Ejecutando consulta MySQL: SELECT * FROM usuarios
# Conectando a la base de datos PostgreSQL...
# Ejecutando consulta PostgreSQL: SELECT * FROM usuarios
```

En este ejemplo:
- `BaseDeDatos` es una clase base abstracta (aunque no usa `ABC`) que define la interfaz para todos los tipos de bases de datos.
- `BaseDeDatosMySQL` y `BaseDeDatosPostgreSQL` son implementaciones concretas.
- `realizar_operacion_en_bd` trabaja con cualquier objeto que cumpla con la interfaz de `BaseDeDatos`.

### Principios de diseño y patrones

La abstracción es un componente clave de varios principios y patrones de diseño importantes:

1. **Principios SOLID**:
   - Principio de Responsabilidad Única (SRP).
   - Principio de Abierto/Cerrado (OCP).
   - Principio de Sustitución de Liskov (LSP).
   - Principio de Segregación de Interfaces (ISP).
   - Principio de Inversión de Dependencias (DIP).

2. **Patrones de Diseño**:
   - Patrón método de fábrica.
   - Patrón fábrica abstracta.
   - Patrón estrategia.
   - Patrón método plantilla.

Veamos una implementación del patrón estrategia:

```python
from abc import ABC, abstractmethod

class EstrategiaOrdenamiento(ABC):
    @abstractmethod
    def ordenar(self, datos):
        pass

class OrdenamientoBurbuja(EstrategiaOrdenamiento):
    def ordenar(self, datos):
        print("Realizando ordenamiento de burbuja")
        return sorted(datos)  # Usamos sorted() de Python por simplicidad

class OrdenamientoRapido(EstrategiaOrdenamiento):
    def ordenar(self, datos):
        print("Realizando ordenamiento rápido")
        return sorted(datos)  # Usamos sorted() de Python por simplicidad

class Ordenador:
    def __init__(self, estrategia):
        self.estrategia = estrategia

    def ordenar(self, datos):
        return self.estrategia.ordenar(datos)

# Uso
datos = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

ordenador_burbuja = Ordenador(OrdenamientoBurbuja())
print(ordenador_burbuja.ordenar(datos))

ordenador_rapido = Ordenador(OrdenamientoRapido())
print(ordenador_rapido.ordenar(datos))

# Salida:
# Realizando ordenamiento de burbuja
# [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
# Realizando ordenamiento rápido
# [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

Este ejemplo del Patrón Estrategia muestra cómo la abstracción nos permite definir una familia de algoritmos, encapsular cada uno y hacerlos intercambiables. La clase `Ordenador` no necesita conocer los detalles de cómo funciona cada algoritmo de ordenamiento; solo sabe que puede llamar al método `ordenar` en cualquier objeto `EstrategiaOrdenamiento`.

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

{{< alert icon="comment" cardColor="white" iconColor="black" textColor="black" >}}
¡Felicitaciones por llegar hasta acá! Espero que este recorrido por el universo de la programación te haya resultado tan interesante como lo fue para mí al escribirlo.

Queremos conocer tu opinión, así que no dudes en compartir tus comentarios, sugerencias y esas ideas brillantes que seguro tenés.

Además, para explorar más allá de estas líneas, date una vuelta por los ejemplos prácticos que armamos para vos. Todo el código y los proyectos los encontrarás en nuestro repositorio de GitHub [learn-software-engineering/examples](https://github.com/learn-software-engineering/examples).

Gracias por ser parte de esta comunidad de aprendizaje. ¡Seguí programando y explorando nuevas areas en este fascinante mundo del software!
{{< /alert >}}

---
