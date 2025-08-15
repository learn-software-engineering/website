---
weight: 12
series: ["Programación: Aprendiendo Orientación a Objetos"]
series_order: 12
title: "Los Cuatro Pilares: Encapsulamiento"
authors:
  - jnonino
description: >
  El encapsulamiento es frecuentemente descripto como el primer pilar de la programación orientada a objetos. Es el mecanismo de agrupar los datos (atributos) y los métodos (funciones) que operan sobre esos datos dentro de una única unidad u objeto. Este concepto también se conoce como ocultamiento de datos, ya que la representación interna del objeto se oculta al mundo exterior.
date: 2024-10-12
tags: ["Programación", "Programación Orientada a Objetos", "OOP", "Los Cuatro Pilares", "Encapsulamiento"]
---

La importancia del encapsulamiento radica en varios aspectos clave:

1. **Protección de datos**: Al controlar el acceso a los datos del objeto a través de métodos, podemos asegurar que los datos se mantengan consistentes y válidos.
2. **Modularidad**: El encapsulamiento permite que los objetos sean autocontenidos, facilitando la comprensión y el mantenimiento del código.
3. **Flexibilidad**: La implementación interna puede ser modificada sin afectar otras partes del código que utilizan el objeto.
4. **Reducción de complejidad**: Al ocultar los detalles del funcionamiento interno, el encapsulamiento reduce la complejidad del sistema desde una perspectiva externa.

### Implementación en Python

Python ofrece varios mecanismos para implementar el encapsulamiento. Exploremos estos con ejemplos:

#### 1. Uso de atributos privados

En Python, podemos crear atributos privados prefijando el nombre del atributo con doble guion bajo (`__`). Esto activa el "name mangling", que hace que el atributo sea más difícil de acceder desde fuera de la clase.

```python
class CuentaBancaria:
    def __init__(self, numero_cuenta, saldo):
        self.__numero_cuenta = numero_cuenta  # Atributo privado
        self.__saldo = saldo  # Atributo privado

    def depositar(self, cantidad):
        if cantidad > 0:
            self.__saldo += cantidad
            return True
        return False

    def retirar(self, cantidad):
        if 0 < cantidad <= self.__saldo:
            self.__saldo -= cantidad
            return True
        return False

    def obtener_saldo(self):
        return self.__saldo

# Uso
cuenta = CuentaBancaria("1234567890", 1000)
print(cuenta.obtener_saldo())  # Salida: 1000
cuenta.depositar(500)
print(cuenta.obtener_saldo())  # Salida: 1500
cuenta.retirar(200)
print(cuenta.obtener_saldo())  # Salida: 1300

# Esto generará un AttributeError
# print(cuenta.__saldo)
```

En este ejemplo:
- `__numero_cuenta` y `__saldo` son atributos privados.
- Proporcionamos métodos públicos (`depositar`, `retirar`, `obtener_saldo`) para interactuar con estos atributos privados.
- El acceso directo a `__saldo` desde fuera de la clase generará una excepción `AttributeError`.

#### 2. Uso de propiedades

El decorador `@property` de Python nos permite definir métodos que pueden ser accedidos como atributos, proporcionando una forma más pythonica de implementar getters y setters.

```python
class Circulo:
    def __init__(self, radio):
        self._radio = radio

    @property
    def radio(self):
        return self._radio

    @radio.setter
    def radio(self, valor):
        if valor > 0:
            self._radio = valor
        else:
            raise ValueError("El radio debe ser positivo")

    @property
    def area(self):
        return 3.14159 * self._radio ** 2

# Uso
circulo = Circulo(5)
print(circulo.radio)  # Salida: 5
print(circulo.area)   # Salida: 78.53975

circulo.radio = 7
print(circulo.radio)  # Salida: 7
print(circulo.area)   # Salida: 153.93791

# Esto generará un ValueError
# circulo.radio = -1
```

En este ejemplo:
- `_radio` es un atributo protegido (el guion bajo simple es una convención para atributos protegidos en Python).
- La propiedad `radio` proporciona acceso de lectura y escritura a `_radio` con validación.
- La propiedad `area` es de solo lectura y se calcula al vuelo.

### Beneficios y mejores prácticas

Los beneficios del encapsulamiento son numerosos:

1. **Mejora de la mantenibilidad**: Los cambios en la implementación interna no afectan al código externo que utiliza la clase.
2. **Mayor seguridad**: Los atributos privados no pueden ser modificados accidentalmente desde fuera de la clase.
3. **Flexibilidad en la implementación**: Puedes cambiar cómo se almacenan o calculan los datos sin cambiar la interfaz pública.
4. **Mejor abstracción**: Los usuarios de la clase no necesitan conocer su funcionamiento interno.

Las mejores prácticas para el encapsulamiento en Python incluyen:

- Usar atributos privados (prefijo de doble guion bajo) para datos que no deben ser accedidos directamente desde fuera de la clase.
- Proporcionar métodos públicos o propiedades para el acceso controlado a los datos internos.
- Usar propiedades en lugar de métodos get/set para un enfoque más propio del paradigma Python.
- Documentar claramente la interfaz pública, incluyendo cualquier efecto secundario de los métodos.

Veamos un ejemplo más complejo que demuestra estas prácticas:

```python
class Empleado:
    def __init__(self, nombre, salario):
        self.__nombre = nombre
        self.__salario = salario
        self.__proyectos = []

    @property
    def nombre(self):
        return self.__nombre

    @property
    def salario(self):
        return self.__salario

    @salario.setter
    def salario(self, valor):
        if valor > 0:
            self.__salario = valor
        else:
            raise ValueError("El salario debe ser positivo")

    def agregar_proyecto(self, proyecto):
        """
        Agrega un proyecto a la lista de proyectos del empleado.

        :param proyecto: cadena que representa el nombre del proyecto
        """
        self.__proyectos.append(proyecto)

    def eliminar_proyecto(self, proyecto):
        """
        Elimina un proyecto de la lista de proyectos del empleado.

        :param proyecto: cadena que representa el nombre del proyecto
        :return: True si el proyecto fue eliminado, False si no se encontró
        """
        if proyecto in self.__proyectos:
            self.__proyectos.remove(proyecto)
            return True
        return False

    @property
    def cantidad_proyectos(self):
        return len(self.__proyectos)

    def __str__(self):
        return f"Empleado: {self.__nombre}, Salario: ${self.__salario}, Proyectos: {self.cantidad_proyectos}"

# Uso
emp = Empleado("Juan Pérez", 50000)
print(emp.nombre)  # Salida: Juan Pérez
print(emp.salario)  # Salida: 50000

emp.agregar_proyecto("Proyecto A")
emp.agregar_proyecto("Proyecto B")
print(emp.cantidad_proyectos)  # Salida: 2

emp.salario = 55000
print(emp)  # Salida: Empleado: Juan Pérez, Salario: $55000, Proyectos: 2

emp.eliminar_proyecto("Proyecto A")
print(emp.cantidad_proyectos)  # Salida: 1

# Esto generará un AttributeError
# print(emp.__proyectos)
```

Este ejemplo demuestra:
- Atributos privados (`__nombre`, `__salario`, `__proyectos`)
- Propiedades para acceso controlado (`nombre`, `salario`, `cantidad_proyectos`)
- Métodos públicos para manipular datos privados (`agregar_proyecto`, `eliminar_proyecto`)
- Documentación clara del comportamiento de los métodos
- Un método `__str__` personalizado para una representación de cadena agradable del objeto

Siguiendo estas prácticas, creamos una clase que es flexible y robusta, encarnando el principio de encapsulamiento.

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
