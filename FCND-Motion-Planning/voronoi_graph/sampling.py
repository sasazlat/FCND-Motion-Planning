from shapely.geometry import Polygon, Point
import numpy as np
from sklearn.neighbors import KDTree


class Poly():

    '''@param coordinates is touple of lat, lon, alt'''
    def __init__(self, coordinates, height):
        self._polygon = Polygon(coordinates)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coordinates(self):
        return list(self._polygon.exterior.coords)[:-1]

    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return (self._polygon.centroid.x, self._polygon.centroid.y)

    def contains(self, point):
        p = Point(point)
        return self._polygon.contains(p)

    def crosses(self, other):
        return self._polygon.crosses(other)


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        obstacle_corners = [(north - d_north, east - d_east),(north - d_north, east + d_east),(north + d_north, east + d_east),(north + d_north, east - d_east)]
        height = alt + d_alt

        p = Poly(obstacle_corners, height)
        polygons.append(p)
    return polygons


class Sampler():
    def __init__(self, data):

        self._polygons = extract_polygons(data)
        self._x_min = np.floor(np.amin(data[:,0] - data[:,3]))
        self._x_max = np.ceil(np.amax(data[:,0] + data[:,3]))

        self._y_min = np.floor(np.amin(data[:,1] - data[:,4]))
        self._y_max = np.ceil(np.amax(data[:,0] + data[:,4]))

        self._z_min = 0
        self._z_max = np.ceil(np.amax(data[:,2]))

        self._centers = np.array([p.center for p in self._polygons])

        self._tree = KDTree(self._centers, metric='euclidean') 

    def sample(self, num_samples=100):
        """
        Returns list of randomly created nodes that do not 
        cross with polygons 
        """
        xvals = np.random.uniform(self._x_min, self._x_max, num_samples)
        yvals = np.random.uniform(self._y_min, self._y_max, num_samples)
        zvals = np.random.uniform(self._z_min, self._z_max, num_samples)
        
        samples = list(zip(xvals, yvals, zvals))
        valid_nodes = []
        for s in samples:
            _, idx = self._tree.query(np.array([s[0], s[1]]).reshape(1,-1))
            p = self._polygons[int(idx)]
            if not p.contains(s) or p.height < s[2]:
                valid_nodes.append(s)
        return valid_nodes

    @property
    def polygons(self):
        return self._polygons

    @property
    def polygons_KDT_centres(self):
        return self._tree

    @property
    def polygons_centers(self):
        return self._centers