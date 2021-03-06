from enum import Enum
from queue import PriorityQueue
from bresenham import bresenham
import csv
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)

def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))


    # add edges to list
    obstacles_centers = []

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1
            obstacle_center = ([north - north_min, east - east_min])
            obstacles_centers.append(obstacle_center)

    graph = Voronoi(obstacles_centers)
    edges = create_vor_edges(grid, obstacles_centers)
    return grid, edges, int(north_min), int(east_min)

def create_vor_edges(grid, obstacles_centers):
    graph = Voronoi(obstacles_centers)
    edges = []
    # check for colision
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        if can_connect(cells, grid):
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            edges.append((p1,p2))
    return edges

def create_nx_graph(vor_edges):
    nx_graph = nx.Graph()
    for e in vor_edges:
        p1 = e[0]
        p2 = e[1]
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        nx_graph.add_edge(p1, p2, weight=dist)
    return nx_graph

# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    NORTH = (-1, 0, 1)
    NORTH_EAST = (-1, 1, np.sqrt(2))
    EAST = (0, 1, 1)
    SOUTH_EAST = (1, 1, np.sqrt(2))
    SOUTH = (1, 0, 1)
    SOUTH_WEST = (1, -1, np.sqrt(2))
    WEST = (0, -1, 1)
    NORTH_WEST = (-1, -1, np.sqrt(2))


    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NORTH_EAST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.SOUTH_EAST)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1:
        valid_actions.remove(Action.SOUTH_WEST)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1:
        valid_actions.remove(Action.NORTH_WEST)

    return valid_actions

def a_star_grid(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

# A* implementation for graph search
def a_star_graph(graph, heuristic, start, goal):

    queue = PriorityQueue()
    queue.put((heuristic(start, goal), start))
    visited = set()

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        visited.add(current_node)

        if current_node == goal:
            print('Found a path.')
            found = True
            break

        else:
            for next_node in graph.adj[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                cost_so_far = (current_cost - heuristic(current_node, goal)) + cost
                new_cost = cost_so_far + heuristic(next_node, goal)
                if next_node not in visited:
                    if next_node in branch:
                        if cost_so_far < branch[next_node][0]:
                            queue.put((new_cost, next_node))
                            branch[next_node] = (cost_so_far, current_node)
                    else:
                        queue.put((new_cost, next_node))
                        branch[next_node] = (cost_so_far, current_node)

    path = []
    if found:
        path = []
        n = goal
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1]

def prune_path_bres(grid, path):

    pruned_path = [p for p in path]

    i = 0
    while i < len(pruned_path) - 2:
        p1, p3 = pruned_path[i], pruned_path[i + 2]
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p3[0]), int(p3[1])))
        if can_connect(cells, grid):
            pruned_path.remove(pruned_path[i + 1])
        else:
            i += 1
    return pruned_path

# Prune the path using Bresenham algorithm in 2D grid
def prune_path_2d(path, grid):
    i = 1
    pruned_path = [path[0]]
    x1, y1 = path[0][0], path[0][1]
    p1 = (x1, y1)
    while i < len(path):
        x2, y2 = path[i][0], path[i][1]
        p2 = (x2, y2)
        if is_safe((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), grid):
            prev_safe_path = p2
            i += 1
        else:
            pruned_path.append(prev_safe_path)
            x1, y1 = prev_safe_path[0], prev_safe_path[1]
            p1 = (x1, y1)
    pruned_path.append(prev_safe_path)
    return pruned_path

def can_connect(cells, grid):
    connected = True
    for cell in cells:
        if np.min(cell) < 0 or cell[0] >= grid.shape[0] or cell[1] >= grid.shape[1]:
            return not connected
        if grid[cell[0], cell[1]] == 1:
            return not connected
    return connected

def colinearity_check(p1, p2, p3):
    det = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    return det == 0

def prune_path_colinearity(grid, path):
    pruned_path = [p for p in path]
    i = 0
    while i < len(pruned_path) - 2:
        p1, p2, p3 = pruned_path[i], pruned_path[i + 1], pruned_path[i + 2]
        if colinearity_check(p1,p2,p3):
            pruned_path.remove(pruned_path[i + 1])
        else:
            i += 1
    return pruned_path

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def read_home_lat_lon(csv_file):
    with open(csv_file, newline='') as f:
        first_line = list(csv.reader(f, delimiter=','))[0]
    lat0 = float(first_line[0].split(" ")[1])
    lon0 = float(first_line[1].split(" ")[1])
    return lat0, lon0

# Return the closest node on graph to the given point in 2D grid
def find_closest_point(graph, point):
    closest_pt = None
    shortest_dis = 10000000
    for n in graph.nodes:
        dis = np.linalg.norm(np.array(n) - np.array(point))
        if dis < shortest_dis:
            closest_pt = n
            shortest_dis = dis
    return closest_pt





#if __name__ == "__main__":
#    lat0, lon0 = read_home_lat_lon("colliders.csv")
#    print (lat0, lon0)