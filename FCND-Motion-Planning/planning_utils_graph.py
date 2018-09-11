import networkx as nx
import numpy as np
import csv

from scipy.spatial import Voronoi
from enum import Enum
from queue import PriorityQueue
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree
from bresenham import bresenham


class Poly:

    def __init__(self, coords, height):
        self._polygon = Polygon(coords)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coords(self):
        return list(self._polygon.exterior.coords)[:-1]

    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return (self._polygon.centroid.x, self._polygon.centroid.y)

    def contains(self, point):
        point = Point(point)
        return self._polygon.contains(point)

    def crosses(self, other):
        return self._polygon.crosses(other)

def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]),
                   (obstacle[1], obstacle[2])]

        height = alt + d_alt

        p = Poly(corners, height)
        polygons.append(p)

    return polygons

class Sampler:

    def __init__(self, data):
        self._polygons = extract_polygons(data)
        self._xmin = np.min(data[:, 0] - data[:, 3])
        self._xmax = np.max(data[:, 0] + data[:, 3])

        self._ymin = np.min(data[:, 1] - data[:, 4])
        self._ymax = np.max(data[:, 1] + data[:, 4])

        self._zmin = 0
        # limit z-axis
        self._zmax = 20

        centers = np.array([p.center for p in self._polygons])
        self._tree = KDTree(centers, metric='euclidean')

    def sample(self, num_samples):
        """Implemented with a k-d tree for efficiency."""
        xvals = np.random.uniform(self._xmin, self._xmax, num_samples)
        yvals = np.random.uniform(self._ymin, self._ymax, num_samples)
        zvals = np.random.uniform(self._zmin, self._zmax, num_samples)
        samples = list(zip(xvals, yvals, zvals))

        pts = []
        for s in samples:
            _, idx = self._tree.query(np.array([s[0], s[1]]).reshape(1, -1))
            p = self._polygons[int(idx)]
            if not p.contains(s) or p.height < s[2]:
                pts.append(s)
        return pts

    @property
    #returns list of polygons
    def polygons(self):
        return self._polygons

    @property
    # returns list of polygons centers
    def polygons_centers(self):
        return self.centers

    @property
    # returns KDTree object of polygons centers
    def tree_polygon_centers(self):
        return self._tree

def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Initialize an empty list for Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # Create a voronoi graph based on location of obstacle centres
    graph = Voronoi(points)
    
    # Check each edge from graph.ridge_vertices for collision
    edges = []
    print("start building edges")
    print(len(graph.ridge_vertices))

    for v in graph.ridge_vertices:

        p1 = tuple(graph.vertices[v[0]])
        p2 = tuple(graph.vertices[v[1]])
        # If any of the vertices is out of grid then skip
        if np.amin(p1) < 0 or np.amin(p2) < 0 or p1[0] >= grid.shape[0] or p1[1] >= grid.shape[1] or p2[0] >= grid.shape[0] or p2[1] >= grid.shape[1]:
            continue

        safe = True
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))

        # Test each pair p1 and p2 for collision using Bresenham
        # If the edge does not hit an obstacle
        # add it to the list

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                return not safe
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                safe = False
                break
        if safe:
            edges.append((p1, p2))
    print("done building edges")
    return grid, edges, int(north_min), int(east_min)

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

# Build a 2D graph
def create_2D_graph(edges):
    G = nx.Graph()
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)
    return G

def find_closest_point(graph, point):
    tree = KDTree(graph.nodes, metric="euclidean")
    _, idx = tree.query([point])
    print (idx[0])
    nodes = list(graph.nodes)
    n = nodes[idx[0][0]]
    print (n)
    return n

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def read_home_lat_lon(csv_file):
    with open(csv_file, newline='') as f:
        first_line = list(csv.reader(f, delimiter=','))[0]
    lat0 = float(first_line[0].split(" ")[1])
    lon0 = float(first_line[1].split(" ")[1])
    return lat0, lon0

def is_safe(p1, p2, grid):
    safe  = True
    cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    for c in cells:
        if grid[c[0], c[1]] == 1:
            safe = False
            break
    return safe


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