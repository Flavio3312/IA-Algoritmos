"""
# -*- coding: utf-8 -*-
# Búsqueda primero en profundidad (DFS) iterativa
# Inteligencia Artificial
# Algoritmos de búsqueda no informada


    
# -*- coding: utf-8 -*-

    
"""


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

