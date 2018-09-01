import numpy as np
import csv
import networkx as nx

from queue import PriorityQueue
from sklearn.neighbors import KDTree
from shapely.geometry import Point, Polygon, LineString


def a_star_graph(graph, h, start, goal):

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
            for next_node in graph[current_node]:
                # get the tuple representation
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node)
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

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def read_home_lat_lon(csv_file):
    with open(csv_file, newline='') as f:
        first_line = list(csv.reader(f, delimiter=','))[0]
    lat0 = float(first_line[0].split(" ")[1])
    lon0 = float(first_line[1].split(" ")[1])
    return lat0, lon0

def can_connect(n1, n2, polygons):
    l = LineString([n1, n2])
    for p in polygons:
        if p.crosses(l) and p.height >= min(n1[2], n2[2]):
            return False
    return True

def create_graph(nodes, k, heuristic, polygons):
    g = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        dist, idxs = tree.query([n1], k)
        i = 0
        for idx in idxs[0]:
            n2 = nodes[idx]
            if n2 == n1:
                i += 1
                continue
            if can_connect(n1, n2, polygons):
                g.add_edge(n1,n2,weight=dist[0][i])
                i += 1
    return g

def closest_point(nodes, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    pts = []
    for n in nodes:
        pts.append((n[0], n[1]))    
    tree = KDTree(pts, metric='euclidean')
    _, idx = tree.query(np.array([current_point[0], current_point[1]]).reshape(1,-1))
    p = nodes[int(idx[0][0])]
    return p