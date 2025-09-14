---
weight: 2
series: ["Matemática para Machine Learning"]
series_order: 2
title: "Proyecto usando Álgebra Lineal: Sistema de Recomendaciones"
description: "Para consolidar los conceptos aprendidos, implementaremos un sistema de recomendaciones completo usando solo álgebra lineal. Este proyecto demuestra cómo el álgebra lineal es fundamental en sistemas de machine learning reales."
authors:
  - jnonino
date: 2025-09-14
tags: ["Inteligencia Artificial", "Aprendizaje Automático", "Machine Learning", "Matemática", "Álgebra Lineal", "Vectores", "Matrices", "Recomendaciones"]
---
{{< katex >}}

{{< lead >}}
Para consolidar los conceptos aprendidos, implementaremos un sistema de recomendaciones completo usando solo álgebra lineal. Este proyecto demuestra cómo el álgebra lineal es fundamental en sistemas de machine learning reales.
{{< /lead >}}

---

## Objetivo del proyecto

Crear un sistema que:
1. Represente usuarios y productos como vectores
2. Calcule similitudes usando productos punto
3. Haga recomendaciones basadas en usuarios similares
4. Visualice los resultados en espacios de menor dimensión

### Implementación Completa

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

class RecommendationSystem:
    """
    Sistema de recomendación basado en álgebra lineal.

    Este proyecto demuestra cómo el álgebra lineal es fundamental
    en sistemas de machine learning reales.
    """

    def __init__(self):
        self.users: Dict[str, np.ndarray] = {}
        self.items: Dict[str, np.ndarray] = {}
        self.ratings_matrix: Optional[np.ndarray] = None
        self.user_similarity_matrix: Optional[np.ndarray] = None
        self.item_features: Optional[np.ndarray] = None
        self.user_names: List[str] = []
        self.item_names: List[str] = []

    def add_user_ratings(self, user_name: str, ratings: Dict[str, float]):
        """
        Agrega las calificaciones de un usuario.

        Args:
            user_name: Nombre del usuario
            ratings: Diccionario {item_name: rating}
        """
        # Convertir ratings a vector (usar 0 para items no calificados)
        if not self.item_names:
            self.item_names = list(ratings.keys())

        rating_vector = np.zeros(len(self.item_names))
        for i, item in enumerate(self.item_names):
            rating_vector[i] = ratings.get(item, 0.0)

        self.users[user_name] = rating_vector
        if user_name not in self.user_names:
            self.user_names.append(user_name)

    def build_ratings_matrix(self):
        """
        Construye la matriz de calificaciones usuarios × items.
        """
        n_users = len(self.user_names)
        n_items = len(self.item_names)

        self.ratings_matrix = np.zeros((n_users, n_items))

        for i, user in enumerate(self.user_names):
            if user in self.users:
                self.ratings_matrix[i] = self.users[user]

        print(f"Matriz de ratings construida: {self.ratings_matrix.shape}")
        return self.ratings_matrix

    def calculate_user_similarity(self, method='cosine'):
        """
        Calcula la matriz de similitud entre usuarios.

        Args:
            method: 'cosine', 'dot_product', o 'euclidean'
        """
        if self.ratings_matrix is None:
            self.build_ratings_matrix()

        n_users = self.ratings_matrix.shape[0]
        self.user_similarity_matrix = np.zeros((n_users, n_users))

        for i in range(n_users):
            for j in range(n_users):
                if method == 'cosine':
                    similarity = self._cosine_similarity(
                        self.ratings_matrix[i],
                        self.ratings_matrix[j]
                    )
                elif method == 'dot_product':
                    similarity = np.dot(
                        self.ratings_matrix[i],
                        self.ratings_matrix[j]
                    )
                elif method == 'euclidean':
                    similarity = -np.linalg.norm(
                        self.ratings_matrix[i] - self.ratings_matrix[j]
                    )

                self.user_similarity_matrix[i, j] = similarity

        return self.user_similarity_matrix

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def recommend_items(self, target_user: str, n_recommendations: int = 3) -> List[Tuple[str, float]]:
        """
        Recomienda items basado en usuarios similares.

        Args:
            target_user: Usuario para quien hacer recomendaciones
            n_recommendations: Número de recomendaciones

        Returns:
            Lista de tuplas (item_name, predicted_rating)
        """
        if target_user not in self.user_names:
            raise ValueError(f"Usuario {target_user} no encontrado")

        if self.user_similarity_matrix is None:
            self.calculate_user_similarity()

        user_idx = self.user_names.index(target_user)
        user_ratings = self.ratings_matrix[user_idx]

        # Encontrar usuarios similares (excluyendo al usuario mismo)
        similarities = self.user_similarity_matrix[user_idx].copy()
        similarities[user_idx] = 0  # Excluir auto-similitud

        # Predecir ratings para items no calificados
        predictions = []

        for item_idx, current_rating in enumerate(user_ratings):
            if current_rating == 0:  # Item no calificado
                # Calcular predicción basada en usuarios similares
                numerator = 0
                denominator = 0

                for other_user_idx, similarity in enumerate(similarities):
                    if similarity > 0 and self.ratings_matrix[other_user_idx, item_idx] > 0:
                        numerator += similarity * self.ratings_matrix[other_user_idx, item_idx]
                        denominator += similarity

                if denominator > 0:
                    predicted_rating = numerator / denominator
                    predictions.append((self.item_names[item_idx], predicted_rating))

        # Ordenar por rating predicho descendente
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_recommendations]

    def analyze_with_pca(self, n_components: int = 2):
        """
        Analiza los datos usando PCA para visualización.

        Args:
            n_components: Número de componentes principales

        Returns:
            Datos proyectados, componentes principales, varianza explicada
        """
        if self.ratings_matrix is None:
            self.build_ratings_matrix()

        # Centrar los datos
        data_centered = self.ratings_matrix - np.mean(self.ratings_matrix, axis=0)

        # Calcular matriz de covarianza
        cov_matrix = np.cov(data_centered.T)

        # Eigendescomposición
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

        # Ordenar por eigenvalores descendentes
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Seleccionar componentes principales
        principal_components = eigenvecs[:, :n_components]

        # Proyectar datos
        projected_data = data_centered @ principal_components

        # Calcular varianza explicada
        explained_variance = eigenvals[:n_components] / np.sum(eigenvals)

        return projected_data, principal_components, explained_variance

    def visualize_users_2d(self):
        """
        Visualiza usuarios en espacio 2D usando PCA.
        """
        projected_data, components, explained_var = self.analyze_with_pca(2)

        plt.figure(figsize=(10, 8))

        # Scatter plot de usuarios
        plt.scatter(projected_data[:, 0], projected_data[:, 1],
                   s=100, alpha=0.7, c='blue')

        # Etiquetar usuarios
        for i, user in enumerate(self.user_names):
            plt.annotate(user, (projected_data[i, 0], projected_data[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)

        plt.xlabel(f'Componente Principal 1 ({explained_var[0]:.1%} varianza)')
        plt.ylabel(f'Componente Principal 2 ({explained_var[1]:.1%} varianza)')
        plt.title('Usuarios en Espacio 2D (PCA)')
        plt.grid(True, alpha=0.3)

        # Mostrar vectores de componentes principales
        plt.arrow(0, 0, components[0, 0]*2, components[0, 1]*2,
                 head_width=0.1, head_length=0.1, fc='red', ec='red',
                 label='PC1')
        plt.arrow(0, 0, components[1, 0]*2, components[1, 1]*2,
                 head_width=0.1, head_length=0.1, fc='green', ec='green',
                 label='PC2')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return projected_data, components, explained_var

    def print_analysis_report(self):
        """
        Imprime un reporte completo del análisis.
        """
        print("=== REPORTE DEL SISTEMA DE RECOMENDACIÓN ===")

        if self.ratings_matrix is not None:
            print(f"\n1. DATOS:")
            print(f"   - Usuarios: {len(self.user_names)}")
            print(f"   - Items: {len(self.item_names)}")
            print(f"   - Matriz de ratings: {self.ratings_matrix.shape}")

            # Estadísticas de la matriz
            non_zero_ratings = np.count_nonzero(self.ratings_matrix)
            total_possible = self.ratings_matrix.size
            sparsity = 1 - (non_zero_ratings / total_possible)

            print(f"   - Ratings dados: {non_zero_ratings}/{total_possible}")
            print(f"   - Sparsity: {sparsity:.1%}")

            print(f"\n2. MATRIZ DE RATINGS:")
            print("   Usuarios × Items:")
            print("   " + "\t".join(f"{item[:8]}" for item in self.item_names))
            for i, user in enumerate(self.user_names):
                ratings_str = "\t".join(f"{r:.1f}" for r in self.ratings_matrix[i])
                print(f"   {user[:8]}\t{ratings_str}")

        if self.user_similarity_matrix is not None:
            print(f"\n3. SIMILITUD ENTRE USUARIOS:")
            print("   " + "\t".join(f"{user[:8]}" for user in self.user_names))
            for i, user in enumerate(self.user_names):
                sim_str = "\t".join(f"{s:.3f}" for s in self.user_similarity_matrix[i])
                print(f"   {user[:8]}\t{sim_str}")

        # PCA Analysis
        projected, components, explained_var = self.analyze_with_pca(2)
        print(f"\n4. ANÁLISIS DE COMPONENTES PRINCIPALES:")
        print(f"   - Componente 1 explica: {explained_var[0]:.1%} de la varianza")
        print(f"   - Componente 2 explica: {explained_var[1]:.1%} de la varianza")
        print(f"   - Total explicado: {sum(explained_var):.1%}")

        print(f"\n   Usuarios en espacio 2D:")
        for i, user in enumerate(self.user_names):
            print(f"   {user}: ({projected[i, 0]:.2f}, {projected[i, 1]:.2f})")

# Crear y probar el sistema de recomendación
def test_recommendation_system():
    """
    Prueba completa del sistema de recomendación.
    """
    print("=== PRUEBA DEL SISTEMA DE RECOMENDACIÓN ===")

    # Crear sistema
    rec_sys = RecommendationSystem()

    # Datos de ejemplo: usuarios y sus calificaciones de películas
    user_ratings = {
        'Ana': {
            'Acción_1': 5.0, 'Comedia_1': 2.0, 'Drama_1': 4.0,
            'Acción_2': 4.0, 'Comedia_2': 1.0, 'Drama_2': 5.0,
            'Acción_3': 0.0, 'Comedia_3': 0.0, 'Drama_3': 0.0
        },
        'Pablo': {
            'Acción_1': 4.0, 'Comedia_1': 3.0, 'Drama_1': 3.0,
            'Acción_2': 5.0, 'Comedia_2': 2.0, 'Drama_2': 4.0,
            'Acción_3': 4.0, 'Comedia_3': 0.0, 'Drama_3': 0.0
        },
        'Carlos': {
            'Acción_1': 1.0, 'Comedia_1': 5.0, 'Drama_1': 2.0,
            'Acción_2': 2.0, 'Comedia_2': 4.0, 'Drama_2': 1.0,
            'Acción_3': 0.0, 'Comedia_3': 5.0, 'Drama_3': 0.0
        },
        'Diana': {
            'Acción_1': 2.0, 'Comedia_1': 1.0, 'Drama_1': 5.0,
            'Acción_2': 1.0, 'Comedia_2': 0.0, 'Drama_2': 4.0,
            'Acción_3': 0.0, 'Comedia_3': 0.0, 'Drama_3': 5.0
        },
        'Elena': {
            'Acción_1': 4.0, 'Comedia_1': 4.0, 'Drama_1': 3.0,
            'Acción_2': 3.0, 'Comedia_2': 4.0, 'Drama_2': 3.0,
            'Acción_3': 0.0, 'Comedia_3': 0.0, 'Drama_3': 0.0
        }
    }

    # Agregar usuarios al sistema
    for user, ratings in user_ratings.items():
        rec_sys.add_user_ratings(user, ratings)

    # Construir matriz y calcular similitudes
    rec_sys.build_ratings_matrix()
    rec_sys.calculate_user_similarity('cosine')

    # Generar reporte
    rec_sys.print_analysis_report()

    # Hacer recomendaciones para cada usuario
    print(f"\n5. RECOMENDACIONES:")
    for user in rec_sys.user_names:
        try:
            recommendations = rec_sys.recommend_items(user, n_recommendations=2)
            print(f"\n   Recomendaciones para {user}:")
            if recommendations:
                for item, rating in recommendations:
                    print(f"   - {item}: {rating:.2f} (predicho)")
            else:
                print(f"   - No hay recomendaciones disponibles")
        except Exception as e:
            print(f"   - Error generando recomendaciones: {e}")

    # Visualización (comentado para evitar issues con display)
    # rec_sys.visualize_users_2d()

    return rec_sys
```

---

## Próximos pasos: optimizaciones y cálculo

El álgebra lineal que acabas de dominar es la base computacional del Machine Learning. En el próximo artículo exploraremos el **cálculo**. Allí aprenderás a:

- **Entender intuitivamente** qué es una derivada y por qué es tan poderosa
- **Calcular gradientes** de funciones de múltiples variables
- **Implementar descenso por gradiente desde cero** en Python
- **Visualizar** cómo los algoritmos *"aprenden"* navegando funciones de costo
- **Optimizar** modelos de Machine Learning usando estos conceptos

Y lo más importante: vas a **ver** el Machine Learning de una forma completamente nueva, entendiendo el motor matemático que lo impulsa.

---

## Reflexión final

Hasta aquí aprendiste:

✅ **Conceptos fundamentales del Álgebra Lineal**
- Vectores como representación de datos
- Matrices como transformaciones
- Espacios vectoriales y transformaciones lineales
- Valores y vectores propios
✅ **Operaciones clave**
- Producto punto para similitud
- Multiplicación matriz-vector para transformaciones
✅ **Aplicaciones prácticas**
- Sistema de recomendación completo
✅ **Herramientas**
- Implementación desde cero para comprensión
- NumPy para optimización
- Visualización de conceptos

**Para recordar**

1. **El álgebra lineal es el lenguaje del ML**: Todo se reduce a vectores y matrices
2. **Las operaciones tienen significado**: No son solo cálculos, representan conceptos
3. **La geometría importa**: Visualizar ayuda a entender
4. **La optimización es crucial**: NumPy vs implementación casera
5. **La teoría guía la práctica**: entender el *"por qué"* antes del *"cómo"*

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
