
# Proyecto de Búsqueda en el Espacio de Estados

## Descripción
Este proyecto implementa y compara dos enfoques de búsqueda para resolver el problema de navegación de un robot en un árbol de decisiones: la búsqueda primero en profundidad (DFS) iterativa y la búsqueda heurística A*. El objetivo es encontrar la posición final del robot en el árbol de decisiones, representada por el nodo 'A'.

## Algoritmos Implementados

### Búsqueda Primero en Profundidad (DFS) Iterativa
El algoritmo DFS iterativo explora exhaustivamente todas las posibles posiciones del robot en el árbol de decisiones. Utiliza una pila para realizar la búsqueda en profundidad y verifica todas las posibles rutas hasta encontrar el objetivo 'A'.

![búsqueda no informada](https://github.com/Flavio3312/IA-Algoritmos/blob/main/busqueda%20no%20informada.png?raw=true)

#### Código
```python
def dfs_iterative(tree, start, objetivo):
    visited = set()  # Track visited nodes
    stack = [start]  # Stack for DFS

    while stack:  # Continue until stack is empty
        node = stack.pop()  # Pop a node from the stack
        if node not in visited:
            visited.add(node)  # Mark node as visited
            print(node)        # Print the current node (for illustration)
            if node == objetivo:
                return visited  # Return visited nodes if objetivo is found
            stack.extend(reversed(tree[node]))  # Add child nodes to stack

    return visited  # Return visited nodes if objetivo is not found

# Ejemplo de uso
tree = {
    'B': ['B1', 'B-1'],
    'B1': ['B2'],
    'B-1': ['B-2'],
    'B2': ['B3'],
    'B-2': ['B-3'],
    'B3': ['B4'],
    'B-3': ['B-4'],
    'B4': ['B5'],
    'B-4': ['B-5'],
    'B5': ['A'],
    'B-5': [],
    'A': []
}

visited_nodes = dfs_iterative(tree, 'B', 'A')

# Mostrar resultado
print("Nodos visitados:", visited_nodes)
```

### Búsqueda Heurística A*
El algoritmo A* utiliza una heurística para guiar la búsqueda hacia el objetivo 'A'. Combina las mejores características del algoritmo de Dijkstra y la búsqueda greedy, utilizando una función heurística basada en la distancia euclidiana.

![búsqueda informada](https://github.com/Flavio3312/IA-Algoritmos/blob/main/busqueda%20informada.png)

#### Código
```python
from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt

def create_node(position: str, g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: str, pos2: str) -> float:
    node_positions = {'B': 0, 'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'A': 6}
    return abs(node_positions[pos2] - node_positions[pos1])

def get_valid_neighbors(node: str) -> List[Tuple[str, float]]:
    neighbors = {
        'B': [('B1', 1)],
        'B1': [('B2', 1)],
        'B2': [('B3', 1)],
        'B3': [('B4', 1)],
        'B4': [('B5', 1)],
        'B5': [('A', 1)]
    }
    return neighbors.get(node, [])

def reconstruct_path(goal_node: Dict) -> List[str]:
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current['position'])
        current = current['parent']
        
    return path[::-1]

def find_path(start: str, goal: str) -> List[str]:
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )
    
    open_list = [(start_node['f'], start)]
    open_dict = {start: start_node}
    closed_set = set()
    
    while open_list:
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        for neighbor_pos, weight in get_valid_neighbors(current_pos):
            if neighbor_pos in closed_set:
                continue
                
            tentative_g = current_node['g'] + weight
            
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
    
    return []

def visualize_path(path: List[str]):
    node_positions = {'B': (0, 0), 'B1': (1, 0), 'B2': (2, 0), 'B3': (3, 0), 'B4': (4, 0), 'B5': (5, 0), 'A': (6, 0)}
    path_coords = [node_positions[node] for node in path]
    
    plt.figure(figsize=(10, 2))
    plt.plot([x for x, y in path_coords], [y for x, y in path_coords], 'b-', linewidth=3, label='Path')
    plt.plot(path_coords, path_coords, 'go', markersize=15, label='Start')
    plt.plot(path_coords[-1], path_coords[-1], 'ro', markersize=15, label='Goal')
    
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("A* Pathfinding Result")
    plt.show()

# Ejemplo de uso 
start = 'B'
goal = 'A'
path = find_path(start, goal)
visualize_path(path)
```

## Comparación de Enfoques

### Ventajas y Dificultades de DFS

**Ventajas:**
- Simplicidad de implementación.
- Menor consumo de memoria.
- Exploración exhaustiva de todos los nodos.

**Dificultades:**
- No garantiza encontrar el camino más corto.
- Puede entrar en ciclos infinitos.
- Orden de exploración no intuitivo.

### Ventajas y Dificultades de A*

**Ventajas:**
- Eficiencia en la búsqueda.
- Garantiza encontrar el camino más corto.
- Flexibilidad en la heurística.

**Dificultades:**
- Mayor consumo de memoria.
- Dependencia de la calidad de la heurística.

