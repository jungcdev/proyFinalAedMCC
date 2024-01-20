import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(puntos, depth=0):
    #inicializamos el caso base
    if len(puntos) == 0:
        return None
    #determinamos el  numero de caracteristicas de los puntos    
    k = len(puntos[0])
    axis = depth % k
    #ordenamos segun el eje actual
    sorted_puntos = sorted(puntos, key=lambda x: x[axis])
    #calculamos el punto medio de la lista
    median = len(puntos) // 2
    #creamos el nodo  y contruimos recursivamente los subarboles
    return Node(
        point=sorted_puntos[median],
        left=build_kdtree(sorted_puntos[:median], depth + 1),
        right=build_kdtree(sorted_puntos[median + 1:], depth + 1)
    )

def kdtree_knn_search(tree, target, k=1, depth=0, best=None, current_neighbors=None):
    # Si el árbol es nulo, retorna los vecinos actuales
    if tree is None:
        return current_neighbors

    # Dimension del espacio
    dim = len(target)
    # Eje de división en base a la profundidad actual
    axis = depth % dim

    if current_neighbors is None:
        current_neighbors = []

    # Iniciamos las  variables para el próximo mejor vecino y próximo subárbol a explorar
    next_best = best
    next_branch = None

    # Comparamos las distancias y actualiza el próximo mejor vecino
    if best is None or np.linalg.norm([tree.point[i] - target[i] for i in range(dim)]) < np.linalg.norm([next_best[i] - target[i] for i in range(dim)]):
        next_best = tree.point
        current_neighbors.append(tree.point)

    # Decidimos  en qué subárbol continuar la búsqueda basándose en la coordenada del eje actual
    if target[axis] < tree.point[axis]:
        next_branch = tree.left
    else:
        next_branch = tree.right

    # Realizamos la búsqueda de vecinos más cercanos de manera recursiva en el próximo subárbol
    current_neighbors = kdtree_knn_search(next_branch, target, k, depth + 1, next_best, current_neighbors)

    # Verificamos si es posible encontrar puntos más cercanos en el otro subárbol
    if abs(target[axis] - tree.point[axis]) < np.linalg.norm([current_neighbors[-1][i] - target[i] for i in range(dim)]) or len(current_neighbors) < k:
        other_branch = tree.right if next_branch is tree.left else tree.left
        current_neighbors = kdtree_knn_search(other_branch, target, k, depth + 1, best, current_neighbors=current_neighbors)

    # Retornamos los k vecinos más cercanos ordenados por distancia al punto objetivo
    return sorted(current_neighbors, key=lambda x: np.linalg.norm([x[i] - target[i] for i in range(dim)]))[:k]


def plot_tree(ax, tree, xmin, xmax, ymin, ymax, depth=0):
    # Verifica si el nodo actual no es nulo
    if tree is not None:
        dim = len(tree.point)
        axis = depth % dim

        if axis == 0:
            ax.plot([tree.point[0], tree.point[0]], [ymin, ymax], color='k', linestyle='-', linewidth=1)
            plot_tree(ax, tree.left, xmin, tree.point[0], ymin, ymax, depth + 1)
            plot_tree(ax, tree.right, tree.point[0], xmax, ymin, ymax, depth + 1)
        else:
            ax.plot([xmin, xmax], [tree.point[1], tree.point[1]], color='k', linestyle='-', linewidth=1)
            plot_tree(ax, tree.left, xmin, xmax, ymin, tree.point[1], depth + 1)
            plot_tree(ax, tree.right, xmin, xmax, tree.point[1], ymax, depth + 1)


def plot_kdtree(tree, puntos, punto_test, k):

    fig, ax = plt.subplots()
    ax.scatter(puntos[:, 0], puntos[:, 1], c='blue', marker='o', label='puntos')
    ax.scatter(punto_test[0], punto_test[1], c='red', marker='x', label='Punto test')

    neighbors = kdtree_knn_search(tree, punto_test, k)
    neighbor_puntos = np.array(neighbors)
    # Dibujamos los vecinos más cercanos en verde
    ax.scatter(neighbor_puntos[:, 0], neighbor_puntos[:, 1], c='green', marker='^', label='Vecinos mas cercanos')

    # Obtenemos los límites actuales de los ejes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Dibijamos la estructura del KD-Tree
    plot_tree(ax, tree, xmin, xmax, ymin, ymax)

    plt.title('KD-Tree con busqueda KNN')
    plt.legend()
    plt.savefig("kdtree.png")
    plt.show()


# Datos de prueba y ejecución
puntos = np.array([[2, 3], [5, 1], [9, 6], [4, 7], [8, 1], [5,6],[7, 2], [6,4]])
punto_test = np.array([5, 5])
k = 2

tree = build_kdtree(puntos)
neighbors = kdtree_knn_search(tree, punto_test, k=k)

print(f"Busqueda: {punto_test}")
print(f"{k} Vecinos cercanos: {neighbors}")
plot_kdtree(tree, puntos, punto_test, k)
