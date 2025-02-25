import numpy as np


def euclidean_norm(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula la distancia Euclidiana entre dos puntos.
    """
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_norm(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula la distancia Manhattan entre dos puntos.
    """
    return np.sum(np.abs(x - y))


def chebyshev_norm(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula la distancia Chebyshev entre dos puntos.
    """
    return np.max(np.abs(x - y))


def minkowski_norm(x: np.ndarray, y: np.ndarray, p: float = 3) -> float:
    """
    Calcula la distancia de Minkowski entre dos puntos para un valor de p dado.

    Parámetros:
    - x: Vector de datos del punto 1.
    - y: Vector de datos del punto 2.
    - p: Parámetro de la norma (p > 0).
         p=1 es Manhattan, p=2 es Euclidiana, p=inf es Chebyshev.

    Retorna:
    - La distancia de Minkowski entre x e y.
    """
    # Validar el valor de p
    if p <= 0:
        raise ValueError("El parámetro p debe ser mayor que 0.")

    # Calcular la norma p de Minkowski
    distancia = np.sum(np.abs(x - y) ** p) ** (1 / p)
    return distancia


def mahalanobis_norm(x: np.ndarray, y: np.ndarray, VI: np.ndarray) -> float:
    """
    Calcula la distancia de Mahalanobis de forma explícita.

    Parámetros:
    - x: Vector de datos del punto 1.
    - y: Vector de datos del punto 2.
    - VI: Matriz inversa de covarianza de los datos.

    Retorna:
    - La distancia de Mahalanobis entre x e y.
    """
    # Convertir a arrays de NumPy y verificar dimensiones
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener las mismas dimensiones.")
    if VI.shape[0] != VI.shape[1] or VI.shape[0] != x.shape[0]:
        raise ValueError("VI debe ser una matriz cuadrada con dimensiones consistentes con x e y.")

    # Vector de diferencia
    delta = x - y

    # Calcular la distancia de Mahalanobis explícitamente
    distancia = np.sqrt(np.dot(np.dot(delta.T, VI), delta))
    return distancia
