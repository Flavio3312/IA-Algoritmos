"""
# -*- coding: utf-8 -*-
# A* Search Algorithm  
# Búsqueda A*
#

# Búsqueda heurística
# Búsqueda informada
# Inteligencia Artificial

# Algoritmos de búsqueda
   
# -*- coding: utf-8 -*-

    
"""



from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt



def create_node(position: str, g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Create a node for the A* algorithm.
    
    Args:
        position: Name of the node (e.g., 'B1', 'B2', 'A')
        g: Cost from start to this node (default: infinity)
        h: Estimated cost from this node to goal (default: 0)
        parent: Parent node (default: None)
    
    Returns:
        Dictionary containing node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: str, pos2: str) -> float:
    """
    Calculate the estimated distance between two nodes.
    """
    # Example heuristic: number of steps between nodes
    node_positions = {'B': 0, 'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'A': 6}
    return abs(node_positions[pos2] - node_positions[pos1])

def get_valid_neighbors(node: str) -> List[Tuple[str, float]]:
    """
    Get all valid neighboring nodes and their weights.
    
    Args:
        node: Current node (e.g., 'B', 'B1')
    
    Returns:
        List of valid neighboring nodes and their weights
    """
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
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current['position'])
        current = current['parent']
        
    return path[::-1]  # Reverse to get path from start to goal

def find_path(start: str, goal: str) -> List[str]:
    """
    Find the optimal path using A* algorithm.
    
    Args:
        start: Starting node (e.g., 'B')
        goal: Goal node (e.g., 'A')
    
    Returns:
        List of nodes representing the optimal path
    """
    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )
    
    # Initialize open and closed sets
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}         # For quick node lookup
    closed_set = set()                      # Explored nodes
    
    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        # Check if we've reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        # Explore neighbors
        for neighbor_pos, weight in get_valid_neighbors(current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue
                
            # Calculate new path cost
            tentative_g = current_node['g'] + weight
            
            # Create or update neighbor
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
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
    
    return []  # No path found

def visualize_path(path: List[str]):
    """
    Visualize the found path.
    """
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
from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt

def create_node(position: str, g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Create a node for the A* algorithm.
    
    Args:
        position: Name of the node (e.g., 'B1', 'B2', 'A')
        g: Cost from start to this node (default: infinity)
        h: Estimated cost from this node to goal (default: 0)
        parent: Parent node (default: None)
    
    Returns:
        Dictionary containing node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: str, pos2: str) -> float:
    """
    Calculate the estimated distance between two nodes.
    """
    # Example heuristic: number of steps between nodes
    node_positions = {'B': 0, 'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'A': 6}
    return abs(node_positions[pos2] - node_positions[pos1])

def get_valid_neighbors(node: str) -> List[Tuple[str, float]]:
    """
    Get all valid neighboring nodes and their weights.
    
    Args:
        node: Current node (e.g., 'B', 'B1')
    
    Returns:
        List of valid neighboring nodes and their weights
    """
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
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current['position'])
        current = current['parent']
        
    return path[::-1]  # Reverse to get path from start to goal

def find_path(start: str, goal: str) -> List[str]:
    """
    Find the optimal path using A* algorithm.
    
    Args:
        start: Starting node (e.g., 'B')
        goal: Goal node (e.g., 'A')
    
    Returns:
        List of nodes representing the optimal path
    """
    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )
    
    # Initialize open and closed sets
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}         # For quick node lookup
    closed_set = set()                      # Explored nodes
    
    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        # Check if we've reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        # Explore neighbors
        for neighbor_pos, weight in get_valid_neighbors(current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue
                
            # Calculate new path cost
            tentative_g = current_node['g'] + weight
            
            # Create or update neighbor
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
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
    
    return []  # No path found

def visualize_path(path: List[str]):
    """
    Visualize the found path.
    """
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
