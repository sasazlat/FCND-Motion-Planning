import numpy as np
import csv


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def read_home_lat_lon(csv_file):
    with open(csv_file, newline='') as f:
        first_line = list(csv.reader(f, delimiter=','))[0]
    lat0 = float(first_line[0].split(" ")[1])
    lon0 = float(first_line[1].split(" ")[1])
    return lat0, lon0
    
