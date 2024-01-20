import math
import matplotlib.pyplot as plt

def distancia_euclidiana(point1, point2):
    squared_differences = [(a - b) ** 2 for a, b in zip(point1, point2)]
    sum_squared_differences = sum(squared_differences)
    euclidean_distance = math.sqrt(sum_squared_differences)
    return euclidean_distance

def KNN(train_data, test_point, k):
    distances = []
    for idx, train_point in enumerate(train_data):
        distance = (idx, distancia_euclidiana(test_point, train_point))
        distances.append(distance)
    # Ordenar distancias de menor a mayor
    distances.sort(key=lambda x: x[1])  
    k_nearest_neighbors = distances[:k]
    
    # Obtener las etiquetas de los k vecinos más cercanos
    labels = []
    for idx, _ in k_nearest_neighbors:
        label = train_data[idx][2]
        labels.append(label)
    return labels


def graficar(train_data, test_point, k, labels):
    train_x = []
    train_y = []

    for point in train_data:
        train_x.append(point[0])
        train_y.append(point[1])
    
    test_x, test_y = test_point[0], test_point[1]
    
    # Asignamos colores específicos a cada clase
    colors = {'A': 'blue', 'B': 'orange'}
    
    # Creamos una lista de colores para cada punto de entrenamiento
    train_colors = []
    for point in train_data:
        color_index = point[2]
        color = colors[color_index]
        train_colors.append(color)
    
    plt.scatter(train_x, train_y, c=train_colors, marker='o', label='Data prueba')
    plt.scatter(test_x, test_y, c=colors[labels[0]], marker='x', label='Test')  
    
    plt.title(f'KNN (k={k}) - Label: {labels[0]}') 
    plt.legend()
    plt.savefig("knn2.png")
    plt.show()


# Datos de prueba y ejecución
train_data = [
    (1, 2, 'A'),
    (2, 3, 'A'),
    (2, 4, 'A'),
    (3, 1, 'B'),
    (4, 2, 'B'),
    (4.1, 1.3, 'B')
]

test_point = (3.5, 1)
k = 3
labels = KNN(train_data, test_point, k)
graficar(train_data, test_point, k, labels)
