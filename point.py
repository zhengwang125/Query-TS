import math
import numpy as np

class Point(object):
    def __init__(self, x, y, traj_id=None):
        self.trajectory_id = traj_id
        self.x = x
        self.y = y

    def __repr__(self):
        return "{0:.8f},{1:.8f}".format(self.x, self.y)

    def get_point(self):
        return self.x, self.y

    def __add__(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError("The other type is not 'Point' type.")
        _add_x = self.x + other.x
        _add_y = self.y + other.y
        return Point(_add_x, _add_y, traj_id=self.trajectory_id)

    def __sub__(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError("The other type is not 'Point' type.")
        _sub_x = self.x - other.x
        _sub_y = self.y - other.y
        return Point(_sub_x, _sub_y, traj_id=self.trajectory_id)

    def __mul__(self, x: float):
        if isinstance(x, float):
            return Point(self.x*x, self.y*x, traj_id=self.trajectory_id)
        else:
            raise TypeError("The other object must 'float' type.")

    def __truediv__(self, x: float):
        if isinstance(x, float):
            return Point(self.x / x, self.y / x, traj_id=self.trajectory_id)
        else:
            raise TypeError("The other object must 'float' type.")

    def distance(self, other: 'Point'):
        return math.sqrt(math.pow(self.x-other.x, 2) + math.pow(self.y-other.y, 2))

    def dot(self, other: 'Point'):
        return self.x * other.x + self.y * other.y

    def as_array(self):
        return np.array((self.x, self.y))


def _point2line_distance(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)
    return np.divide(np.abs(np.linalg.norm(np.cross(end - start, start - point))),
                     np.linalg.norm(end - start))