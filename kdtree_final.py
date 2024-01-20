import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    #inicializamos el caso base
    if len(points) == 0:
        return None
    #determinamos numero de caracteristicas de los puntos   
    k = len(points[0])
    axis = depth % k
    #ordenamos segun el eje actual
    sorted_points = sorted(points, key=lambda x: x[axis])
    median = len(points) // 2
    #creamos el nodo  y contruimos recursivamente los subarboles
    return Node(
        point=sorted_points[median],
        left=build_kdtree(sorted_points[:median], depth + 1),
        right=build_kdtree(sorted_points[median + 1:], depth + 1)
    )

def kdtree_knn_search(tree, target, k=1, depth=0, best=None, current_neighbors=None):
    # Si el árbol es None, retorna el mejor vecino 
    if tree is None:
        return current_neighbors

    # Dimension del espacio
    dim = len(target)
    # Eje de división en base a la profundidad actual
    axis = depth % dim

    if current_neighbors is None:
        current_neighbors = []

    # Iniciamos las variables para el próximo mejor vecino y próximo subárbol a explorar
    next_best = best
    next_branch = None

    # Comparamos  las distancias y actualiza el próximo mejor vecino
    if best is None or np.linalg.norm(tree.point - target) < np.linalg.norm(next_best - target):
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
    if abs(target[axis] - tree.point[axis]) < np.linalg.norm(current_neighbors[-1] - target) or len(current_neighbors) < k:
        other_branch = tree.right if next_branch is tree.left else tree.left
        current_neighbors = kdtree_knn_search(other_branch, target, k, depth + 1, best, current_neighbors=current_neighbors)

    # Retornamos los k vecinos más cercanos ordenados por distancia al punto objetivo
    return sorted(current_neighbors, key=lambda x: np.linalg.norm(x - target))[:k]

def plot_tree(ax, tree, xmin, xmax, ymin, ymax, depth=0):
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

def plot_kdtree(tree, points, query_point, k):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c='blue', marker='o', label='Puntos')
    ax.scatter(query_point[0], query_point[1], c='red', marker='x', label='Test point')

    neighbors = kdtree_knn_search(tree, query_point, k)
    neighbor_points = np.array(neighbors)
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], c='green', marker='^', label='Vecinos mas cercanos')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    plot_tree(ax, tree, xmin, xmax, ymin, ymax)

    plt.title('KD-Tree con busqueda KNN')
    plt.legend()
    plt.savefig("kdtree_final.png")
    plt.show()

def plot_knn(points, query_point, neighbors, labels):
    fig, ax = plt.subplots()
    unique_labels = list(set(labels))
    label_colors = ['red', 'blue', 'green'] 

    for i, label in enumerate(unique_labels):
        label_points = points[labels == label]
        ax.scatter(label_points[:, 0], label_points[:, 1], c=label_colors[i], marker='o', label=f'Clase: {label}')

    ax.scatter(query_point[0], query_point[1], c='red', marker='x', label='Test point')
    neighbor_points = np.array(neighbors)
    ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], c='yellow', marker='^', label='Vecinos más cercanos')

    # Dibujar el círculo alrededor de los vecinos más cercanos
    knn_circle = plt.Circle((query_point[0], query_point[1]), np.linalg.norm(neighbor_points[-1] - query_point), fill=False, color='blue', linestyle='dashed', linewidth=2)
    ax.add_patch(knn_circle)

    plt.title('Busqueda KNN')
    plt.legend()
    plt.savefig("knn_final.png")
    plt.show()

def keyword_frequency_descriptor(text, keywords):
    features = [text.lower().count(keyword) for keyword in keywords]
    return features
# Ruta al archivo CSV
csv_file_path = "imdb_dataset_test.csv" 

# Lista para almacenar las reseñas y etiquetas
dataset = []

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if len(row) == 2:
            review = row[0].strip()
            label = row[1].strip()
            dataset.append((review, label))

keywords = ["great","slower","favorite", "fantastic","good","wonderful","terrible","boring","laughter", "disappointment", "heartwarming", "sadness", "delight", "lackluster", "predictable", "joy"]

X = []
y = []

for review, label in dataset:
    features = keyword_frequency_descriptor(review, keywords)
    X.append(features)
    y.append(label)

print(X)
X = np.array(X)
y = np.array(y)

# Aplicar PCA para reducción de dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca)
# Dividir el conjunto de datos en entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train, y_train = X_pca, y
# Construimos el KD-Tree con el conjunto de entrenamiento
tree = build_kdtree(X_train)

#Pruebas con un nuevo registro....
#new_review = "This movie is absolutely terrible and boring."
new_review ="This a fantastic movie of three prisoners who become famous. One of the actors is george clooney and I'm not a fan but this roll is not bad. Another good thing about the movie is the soundtrack (The man of constant sorrow)."
#new_review ="The cast played Shakespeare.<br /><br />Shakespeare lost.<br /><br />I appreciate that this is trying to bring Shakespeare to the masses, but why ruin something so good.<br /><br />Is it because 'The Scottish Play' is my favorite Shakespeare? I do not know. What I do know is that a certain Rev Bowdler (hence bowdlerization) tried to do something similar in the Victorian era.<br /><br />In other words, you cannot improve perfection.<br /><br />I have no more to write but as I have to write at least ten lines of text (and English composition was never my forte I will just have to keep going and say that this movie,"
# Extraer las características de la nueva reseña
new_review_features = keyword_frequency_descriptor(new_review, keywords)

# Reducir dimensiones con PCA
new_review_pca = pca.transform([new_review_features])

# Realizar la búsqueda de vecinos más cercanos en el KD-Tree para la nueva reseña
new_review_neighbors = kdtree_knn_search(tree, new_review_pca[0], k=3) 
new_review_neighbor_indices = [np.where((X_train == neighbor).all(axis=1))[0][0] for neighbor in new_review_neighbors]
new_review_neighbor_labels = y_train[new_review_neighbor_indices]
new_review_predicted_label = max(set(new_review_neighbor_labels), key=list(new_review_neighbor_labels).count)

# Etiqueta real para la nueva reseña
true_label = "positive"  

# Imprimir la predicción para la nueva reseña
print(f"Nueva resenia: '{new_review}'")
print(f"Prediccion para la nueva resenia: {new_review_predicted_label}")

# Calcular el accuracy para la nueva reseña
accuracy = accuracy_score([new_review_predicted_label], [true_label])
print(f"Accuracy para la nueva resenia: {accuracy}")

plot_kdtree(tree, X_train, new_review_pca[0], k=3) 
plot_knn(X_train, new_review_pca[0], new_review_neighbors, y_train)

