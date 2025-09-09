---
weight: 3
series: ["Fundamentos de IA para Programadores"]
series_order: 3
title: "Ejemplo 2 - Sistema de Clasificación"
description: "Vamos a crear un clasificador de texto básico usando técnicas estadísticas simples. Esto te muestra los conceptos fundamentales detrás de algoritmos más complejos como Naive Bayes."
authors:
  - jnonino
date: 2025-08-26
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Clasificación"]
---
{{< katex >}}

{{< lead >}}
Ahora vamos a crear un clasificador de texto básico usando técnicas estadísticas simples. Esto te muestra los conceptos fundamentales detrás de algoritmos más complejos como <i>Naive Bayes</i>.
{{< /lead >}}

---

Imaginá que trabajás en una empresa de software y cada día llegan cientos de tickets de soporte técnico. Algunos reportan errores (bugs), otros solicitan nuevas funcionalidades, y otros simplemente piden ayuda. Manualmente clasificar cada ticket tomaría horas de trabajo.

¿Sería genial si pudiéramos enseñarle a una computadora a leer estos tickets y clasificarlos automáticamente? Este es exactamente el tipo de problema que resuelve el **aprendizaje automático (machine learning)**.

Para resolver este problema, implementaremos un [**Clasificador Bayesiano Ingenuo (Naive Bayes Classifier)**](https://es.wikipedia.org/wiki/Clasificador_bayesiano_ingenuo), uno de los algoritmos más elegantes y comprensibles del aprendizaje automático (*machine learning*). ¿Por qué es perfecto para empezar?

- Es **intuitivo**: funciona de manera similar a como los humanos categorizamos
- Es **eficiente**: requiere relativamente pocos datos de entrenamiento
- Es **interpretable**: podemos entender exactamente por qué toma cada decisión

---

## Fundamentos matemáticos: el teorema de Bayes

Antes de sumergirnos en el código, entendamos la base matemática. El [teorema de Bayes](https://es.wikipedia.org/wiki/Teorema_de_Bayes) es una regla matemática que nos para invertir [probabilidades condicionadas](https://es.wikipedia.org/wiki/Probabilidad_condicionada), permitiendonos encontrar la probabilidad de una causa dado su efecto.

La probabilidad condicionada es una medida de la probabilidad de que ocurra un evento, dado que ya se sabe que ha ocurrido otro suceso. Si el suceso de interés es \(A\)  y se sabe o se supone que ha ocurrido el suceso \(B\), la probabilidad condicional de \(A\) dado \(B\), suele escribirse como:

$$P(A|B)$$

Aunque las probabilidades condicionales pueden proporcionar información muy útil, a menudo se cuenta con información limitada. Por lo tanto, puede ser útil invertir la probabilidad condicional utilizando el *teorema de Bayes*:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

En nuestro contexto:
- \(A\) = la categoría (*BUG*, *FEATURE*, *SUPPORT*)
- \(B\) = el texto del ticket

Entonces queremos calcular:

$$P(\text{categoría}|\text{texto})$$

Es decir, la probabilidad de la categoría de un ticket dado la probabilidad de un determinado texto.

Para clasificación, podemos simplificar la fórmula a:

$$P(\text{categoría}|\text{texto}) \propto P(\text{categoría}) \cdot P(\text{texto}|\text{categoría})$$

Siendo:
- \(P(\text{categoría}|\text{texto})\): la probabilidad de que un ticket corresponda a una determinada categoría dado un texto.
- \(P(\text{categoría})\): la probabilidad de que un ticket sea clasificado como de una determinada categoría.
- \(P(\text{texto}|\text{categoría})\): la probabilidad de que el ticket contenta un determinado texto si pertenece a una categoría.

En otras palabras, queremos conocer la probabilidad de que un ticket sea una categoría. Para ello, necesitamos saber la probabilidad de que dicha categoría aplique a un ticket, y la probabilidad con la que determinadas palabras aparecen en los tickets de una determinada categoría.

---

## Estructura básica del sistema

Comenzaremos definiendo la estructura del sistema, por lado, crearemos una clase que actuará como clasificador y una función *main* que será la encargada de entrenarlo y de enviarle nuevos tickets para determinar su categoría.

```python
from collections import defaultdict, Counter

class ClasificadorTextoBasico:
    """
    Clasificador de texto usando probabilidades bayesianas básicas.
    Útil para clasificar emails, reseñas, tickets de soporte, etc.
    """

    def __init__(self):
        # Almacena las frecuencias de palabras por categoría
        self.palabras_por_categoria = defaultdict(Counter)
        # Almacena cuántos tickets hay por categoría
        self.tickets_por_categoria = defaultdict(int)
        # Lista de todas las categorías conocidas
        self.categorias = set()
        # Vocabulario total (todas las palabras únicas)
        self.vocabulario = set()

    def entrenar(self, datos):
        """
        Entrena el clasificador con ejemplos de texto etiquetados.

        Args:
            datos (list): Lista de tuplas cuyo primer valor es el contenido
                          del ticket y el segundo valor, la categoría
                          correspondiente.
        """
        print(f"Entrenando clasificador con {len(datos)} ejemplos...")
        pass


# Ejemplo práctico: Clasificador de tickets de soporte
if __name__ == "__main__":
    # Datos de entrenamiento simulando tickets de soporte técnico
    # Cada elemento de la lista contiene una tupla compuesta por
    # la descripción del ticket y la categoría a la que pertenece.
    datos_de_entrenamiento = [
        ("La aplicación se cierra inesperadamente al hacer click en enviar", "BUG"),
        ("Error 500 al intentar subir archivo grande", "BUG"),
        ("El botón de guardar no funciona en Firefox", "BUG"),
        ("Pantalla en blanco después de iniciar sesión", "BUG"),
        ("Los datos no se actualizan correctamente en la tabla", "BUG"),
        ("Mensaje de error extraño al procesar el pago", "BUG"),
        ("Sería genial poder exportar reportes a Excel", "FEATURE"),
        ("Necesitamos filtros avanzados en el listado de productos", "FEATURE"),
        ("Propongo agregar notificaciones push para mensajes", "FEATURE"),
        ("Falta la opción de cambiar el idioma de la interfaz", "FEATURE"),
        ("Queremos integración con Google Calendar", "FEATURE"),
        ("Deberíamos tener dashboard personalizable para cada usuario", "FEATURE"),
        ("Cómo puedo cambiar mi contraseña", "SUPPORT"),
        ("No entiendo cómo funciona el sistema de permisos", "SUPPORT"),
        ("Necesito ayuda para configurar mi perfil", "SUPPORT"),
        ("Dónde encuentro las estadísticas de ventas", "SUPPORT"),
        ("Instrucciones para conectar con la API", "SUPPORT"),
        ("Tutorial para usar las funciones avanzadas", "SUPPORT")
    ]

    # Crear y entrenar el clasificador
    clasificador = ClasificadorTextoBasico()
    clasificador.entrenar(datos_de_entrenamiento)
```

El clasificador almacena los siguientes datos:
- `palabras_por_categoria`: automáticamente cuenta frecuencias de palabras que aparecen por categoría
- `tickets_por_categoria`: la cantidad de tickets conocidos en cada categoría. Necesario para calcular \(P(\text{categoría})\)
- `categorias`: listado de las categorías conocidas
- `vocabulario`: conjunto de todas las palabras únicas que hemos visto

---

## Preprocesamiento del texto

Antes de continuar, es necesario incluir una función para preprocesar el texto. Los textos que escribimos pueden tener muchas variaciones, necesitamos convertirlo a una forma estándar, sin distinciones entre mayúsculas y minúsculas, signos de puntuación, tildes, etcétera.

```python
def preprocesar_texto(self, texto):
    """
    Limpia y tokeniza el texto de entrada.

    Args:
        texto (str): Texto a procesar

    Returns:
        list: Lista de palabras limpias en minúsculas
    """
    # Convertir a minúsculas
    texto = texto.lower()

    # Separa las letras de sus diacríticos
    texto = unicodedata.normalize('NFD', texto)

    # Elimina los caracteres de tipo marca diacrítica (Mn), es decir, las tildes y diéresis
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')

    # Tokenizar (dividir en palabras)
    palabras = texto.split()

    # Filtrar palabras muy cortas
    palabras = [p for p in palabras if len(p) >= 3]

    return palabras
```

**¿Por qué estos pasos?**
1. *Minúsculas*: "Error" y "error" deben tratarse igual
2. *Sin puntuación*: Nos enfocamos en las palabras, no en la estructura
3. *Filtrar palabras cortas*: "el", "de", "un" aportan poco valor discriminativo

---

## Entrenando el sistema

El siguiente paso, es entrenar el sistema con datos conocidos para que aprenda a cuando aplicar una categoría u otra dependiendo del texto recibido.
Durante el entrenamiento, el algoritmo *"memoriza"* qué palabras aparecen frecuentemente en cada categoría, y las agrupa según la categoría dada.
Por ejemplo, si vemos *"error"* en 10 tickets con la categoría *BUG* y solo 1 perteneciente a *FEATURE*, el algoritmo aprende que *"error"* es una fuerte señal de *BUG*.

```python
def entrenar(self, datos):
    """
    Entrena el clasificador con ejemplos de texto etiquetados.

    Args:
        datos (list): Lista de tuplas cuyo primer valor es el contenido
                      del ticket y el segundo valor, la categoría
                      correspondiente.
    """
    for texto, categoria in datos:
        palabras = self.preprocesar_texto(texto)

        # Actualizar contadores
        self.categorias.add(categoria)
        self.tickets_por_categoria[categoria] += 1

        # Contar frecuencia de cada palabra en esta categoría
        for palabra in palabras:
            self.palabras_por_categoria[categoria][palabra] += 1
            self.vocabulario.add(palabra)

    print(f"Entrenamiento completado:")
    print(f"   * Categorías: {sorted(self.categorias)}")
    print(f"   * Vocabulario: {len(self.vocabulario)} palabras únicas")
    for cat in sorted(self.categorias):
        print(f"   * '{cat}': {self.tickets_por_categoria[cat]} tickets")
```

Hasta aquí obtendremos esto cuando ejecutamos el programa:
```bash
Entrenando clasificador con 18 ejemplos...
Entrenamiento completado:
   * Categorías: ['BUG', 'FEATURE', 'SUPPORT']
   * Vocabulario: 80 palabras únicas
   * 'BUG': 6 tickets
   * 'FEATURE': 6 tickets
   * 'SUPPORT': 6 tickets
```

---

## Cálculo de probabilidades

Aquí presentamos el corazón del algoritmo, realizaremos los cálculos de probabilidades por palabra que luego servirán para clasificar tickets.

```python
def calcular_probabilidad_palabra(self, palabra, categoria):
    """
    Calcula P(palabra|categoria) usando suavizado de Laplace.
    El suavizado evita probabilidades de 0 para palabras no vistas.

    Args:
        palabra (str): Palabra a evaluar
        categoria (str): Categoría a evaluar

    Returns:
        float: Probabilidad de la palabra dada la categoría
    """
    # Frecuencia de la palabra en esta categoría
    frecuencia_palabra = self.palabras_por_categoria[categoria][palabra]

    # Total de palabras en esta categoría
    total_palabras_categoria = sum(self.palabras_por_categoria[categoria].values())

    # Suavizado de Laplace: sumamos 1 al numerador y |vocabulario| al denominador
    # Esto evita probabilidades de 0 para palabras nuevas
    probabilidad = (frecuencia_palabra + 1) / (total_palabras_categoria + len(self.vocabulario))

    return probabilidad
```

**¿Qué es el suavizado de Laplace?**

Con las [técnicas de suavizado](https://es.wikipedia.org/wiki/Suavizado_de_n-gramas) intentamos evitar las probabilidades cero producidas por palabras no vistas.

Sin suavizado, si una palabra nunca apareció en una categoría, su probabilidad sería \(0\), y todo el cálculo se volvería \(0\).

Con la técnica de suavizado de Laplace, agregamos \(1\) al numerador y el tamaño del vocabulario al denominador. Esto da una probabilidad pequeña pero no nula a palabras no vistas.

$$P(\text{palabra}|\text{categoría}) = \frac{\text{frecuencia} + 1}{\text{palabras en categoría} + \text{tamaño vocabulario}}$$

---

## Clasificando nuevos tickets

Ahora sí llegó el momento de inyectar nuevos tickets y dejar que el algoritmo los clasifique utilizando las probabilidades calculadas antes.

```python
def clasificar(self, texto):
    """
    Clasifica un texto usando el teorema de Bayes.

    P(categoria|texto) ∝ P(categoria) * ∏P(palabra|categoria)

    Args:
        texto (str): Texto a clasificar

    Returns:
        dict: Probabilidades por categoría
    """
    palabras = self.preprocesar_texto(texto)

    # Calculamos log-probabilidades para evitar underflow
    # (multiplicar muchas probabilidades pequeñas da números muy pequeños)
    log_probabilidades = {}

    for categoria in sorted(self.categorias):
        # P(categoria) = tickets_categoria / total_tickets
        total_tickets = sum(self.tickets_por_categoria.values())
        prob_categoria = self.tickets_por_categoria[categoria] / total_tickets

        # Empezamos con log(P(categoria))
        log_prob = math.log(prob_categoria)

        # Multiplicamos por P(palabra|categoria) para cada palabra
        # En log-space: log(a*b) = log(a) + log(b)
        for palabra in palabras:
            prob_palabra = self.calcular_probabilidad_palabra(palabra, categoria)
            log_prob += math.log(prob_palabra)

        log_probabilidades[categoria] = log_prob

    # Convertir de vuelta a probabilidades normales
    # Usamos el truco: exp(log_prob - max_log_prob) para estabilidad numérica
    max_log_prob = max(log_probabilidades.values())
    probabilidades = {}

    for categoria, log_prob in log_probabilidades.items():
        probabilidades[categoria] = math.exp(log_prob - max_log_prob)

    # Normalizar para que sumen 1
    total = sum(probabilidades.values())
    for categoria in probabilidades:
        probabilidades[categoria] /= total

    return probabilidades
```

Te estarás preguntando, **¿por qué usar logaritmos?**, lo que sucede es que multiplicar muchas probabilidades pequeñas puede resultar en números extremadamente pequeños, esto puede conducir a lo que en computación se denomina [***underflow***](https://en.wikipedia.org/wiki/Arithmetic_underflow). Esto significa que la computadora no puede representar un número tan pequeño y se pueden producir errores inesperados o excepciones.

Los logaritmos transforman multiplicaciones en sumas y así evitamos el *underflow*.

$$\log(a \times b \times c) = \log(a) + \log(b) + \log(c)$$

### Probando el clasificador

Agregamos el siguiente código a la función *main* para ejecutar el clasificador en tickets diferentes utilizados en el entrenamiento.

```python
# Probar con tickets nuevos
print("\n\nPROBANDO CLASIFICADOR CON TICKETS NUEVOS:")
print("=" * 60)

tickets_prueba = [
    "La pagina queda en blanco cuando cargo muchos productos",
    "Me gustaría poder ordenar la lista por fecha de creación",
    "No sé cómo resetear mi cuenta de usuario"
]

for i, ticket in enumerate(tickets_prueba, 1):
    print(f"\n#{i}: '{ticket}'")

    resultados = clasificador.clasificar(ticket)

    # Mostrar probabilidades ordenadas
    for categoria, prob in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        print(f"   {categoria}: {prob:.1%}")

    # Mostrar predicción final
    mejor_categoria = max(resultados.items(), key=lambda x: x[1])
    print(f"   -> CLASIFICACIÓN: {mejor_categoria[0]} ({mejor_categoria[1]:.1%} confianza)")
```

Obtenemos la siguiente respuesta:

```bash
PROBANDO CLASIFICADOR CON TICKETS NUEVOS:
============================================================

#1: 'La pagina queda en blanco cuando cargo muchos productos'
   BUG: 41.9%
   FEATURE: 32.6%
   SUPPORT: 25.5%
   -> CLASIFICACIÓN: BUG (41.9% confianza)

#2: 'Me gustaría poder ordenar la lista por fecha de creación'
   FEATURE: 42.5%
   SUPPORT: 31.2%
   BUG: 26.4%
   -> CLASIFICACIÓN: FEATURE (42.5% confianza)

#3: 'No sé cómo resetear mi cuenta de usuario'
   SUPPORT: 55.1%
   FEATURE: 28.5%
   BUG: 16.4%
   -> CLASIFICACIÓN: SUPPORT (55.1% confianza)
```

## Entendiendo las decisiones (interpretabilidad)

Una ventaja clave de **Naive Bayes** es que podemos inspeccionar lo qué aprendió:

```python
def palabras_mas_representativas(self, n=10):
    """
    Encuentra las palabras que mejor distinguen entre categorías.
    Útil para entender lo que aprendió el modelo.
    """
    print(f"\nTOP {n} PALABRAS MÁS REPRESENTATIVAS POR CATEGORÍA:")
    print("=" * 60)

    for categoria in sorted(self.categorias):
        # Calculamos la "representatividad" de cada palabra
        # usando la frecuencia relativa en esta categoría vs otras
        scores = {}

        for palabra in sorted(self.vocabulario):
            freq_categoria = self.palabras_por_categoria[categoria][palabra]
            total_categoria = sum(self.palabras_por_categoria[categoria].values())

            # Frecuencia en otras categorías
            freq_otras = 0
            total_otras = 0
            for otra_cat in self.categorias:
                if otra_cat != categoria:
                    freq_otras += self.palabras_por_categoria[otra_cat][palabra]
                    total_otras += sum(self.palabras_por_categoria[otra_cat].values())

            if total_otras > 0:
                # Relación de frecuencias relativas
                rel_freq_cat = (freq_categoria + 1) / (total_categoria + len(self.vocabulario))
                rel_freq_otras = (freq_otras + 1) / (total_otras + len(self.vocabulario))
                scores[palabra] = rel_freq_cat / rel_freq_otras

        # Mostrar palabras mas representativas
        top_palabras = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:n]

        print(f"\n  {categoria.upper()}:")
        for palabra, score in top_palabras:
            print(f"   • {palabra:<15} (score: {score:.2f})")
```

Esto nos muestra qué palabras son más características de cada categoría.

Desde la función `main` consultamos las palabras más representativas.

```python
# Mostrar palabras más representativas
clasificador.palabras_mas_representativas(15)
```

Obteniendo:

```bash
TOP 15 PALABRAS MÁS REPRESENTATIVAS POR CATEGORÍA:
============================================================

  BUG:
   * error           (score: 3.83)
   * actualizan      (score: 2.55)
   * aplicacion      (score: 2.55)
   * archivo         (score: 2.55)
   * blanco          (score: 2.55)
   * boton           (score: 2.55)
   * cierra          (score: 2.55)
   * click           (score: 2.55)
   * correctamente   (score: 2.55)
   * datos           (score: 2.55)
   * despues         (score: 2.55)
   * enviar          (score: 2.55)
   * extrano         (score: 2.55)
   * firefox         (score: 2.55)
   * grande          (score: 2.55)

  FEATURE:
   * agregar         (score: 2.39)
   * avanzados       (score: 2.39)
   * cada            (score: 2.39)
   * calendar        (score: 2.39)
   * dashboard       (score: 2.39)
   * deberiamos      (score: 2.39)
   * excel           (score: 2.39)
   * exportar        (score: 2.39)
   * falta           (score: 2.39)
   * filtros         (score: 2.39)
   * genial          (score: 2.39)
   * google          (score: 2.39)
   * idioma          (score: 2.39)
   * integracion     (score: 2.39)
   * interfaz        (score: 2.39)

  SUPPORT:
   * como            (score: 4.02)
   * avanzadas       (score: 2.68)
   * ayuda           (score: 2.68)
   * conectar        (score: 2.68)
   * configurar      (score: 2.68)
   * contrasena      (score: 2.68)
   * donde           (score: 2.68)
   * encuentro       (score: 2.68)
   * entiendo        (score: 2.68)
   * estadisticas    (score: 2.68)
   * funciones       (score: 2.68)
   * instrucciones   (score: 2.68)
   * necesito        (score: 2.68)
   * perfil          (score: 2.68)
   * permisos        (score: 2.68)
```

## Análisis y extensiones

¿Por qué se llama ***Naive (Ingenuo)***? Porque asume que todas las palabras son independientes entre sí. En realidad, sabemos que esto no es cierto, *"no funciona"* tiene un significado diferente que *"no"* y *"funciona"* por separado.

Sin embargo, esta simplicidad es también su fortaleza:
- **Eficiencia**: requiere menos datos y cómputo
- **Robustez**: funciona bien incluso cuando la suposición no se cumple perfectamente
- **Interpretabilidad**: fácil de entender y depurar

| **Ventajas**                     | **Limitaciones**                        |
| -------------------------------- | --------------------------------------- |
| Rápido de entrenar y clasificar  | Asume independencia de palabras         |
| Funciona bien con pocos datos    | Sensible a características irrelevantes |
| Maneja bien múltiples categorías | Puede dar probabilidades mal calibradas |
| Probabilidades interpretables    | No captura orden de palabras            |
| Resistente al sobreajuste        |                                         |

## Extensiones posibles

Una vez que domines este clasificador básico, puedes explorar:

1. [**N-gramas**](https://es.wikipedia.org/wiki/N-grama): Considerar procesar grupos de dos, tres o \(N\) palabras
2. [**TF-IDF**](https://es.wikipedia.org/wiki/Tf-idf): Pesar palabras por su importancia relativa
3. **Validación cruzada**: Evaluar mejor el rendimiento
5. **Características adicionales**: Longitud del texto, mayúsculas, etc.

---

## ¿Qué aprendemos de este ejemplo?

1. **El preprocesamiento es crucial**: Limpiar y normalizar el texto afecta directamente la calidad del modelo.

2. **Teorema de Bayes en acción**: Combinamos la probabilidad previa P(categoría) con la evidencia P(palabras|categoría).

3. **Suavizado de Laplace**: Técnica esencial para manejar palabras que no vimos durante el entrenamiento.

4. **Log-probabilidades**: Truco numérico para evitar underflow al multiplicar muchas probabilidades pequeñas.

5. **Interpretabilidad**: Podemos entender qué palabras son más importantes para cada categoría.

---

## Código completo

Aquí tienes el código completo del sistema. También puedes encontrarlo en el repositorio de ejemplos haciendo click en el siguiente [enlace](https://github.com/learn-software-engineering/examples/blob/main/ai/module1/sistema_de_clasificacion/main.py).

{{< github repo="learn-software-engineering/examples" showThumbnail=false >}}

---

## Conclusión

¡Has construido tu primer clasificador de `Machine Learning` desde cero! Este clasificador ***Bayesiano Ingenuo (Naive Bayes)*** demuestra algunos conceptos fundamentales:

- **Aprendizaje supervisado**: Aprender de ejemplos etiquetados
- **Probabilidad**: Cuantificar incertidumbre
- **Generalización**: Aplicar lo aprendido a casos nuevos
- **Interpretabilidad**: Entender las decisiones del modelo

Aunque se vea simple, este tipo de clasificador se usa en aplicaciones reales como filtros de spam, análisis de sentimientos, clasificación de documentos, moderación de contenido, entre otros.

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
