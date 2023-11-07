from typing import Tuple
import math
from point import Point, _point2line_distance

class Segment(object):
    eps = 1e-12
    
    def __init__(self, start_point: Point, end_point: Point, traj_id: int = None, cluster_id: int = -1):
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id
        self.cluster_id = cluster_id

    def set_cluster(self, cluster_id: int):
        self.cluster_id = cluster_id

    def pair(self) -> Tuple[Point, Point]:
        return self.start, self.end

    @property
    def length(self):
        return self.end.distance(self.start)

    def perpendicular_distance(self, other: 'Segment'):
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)

    def parallel_distance(self, other: 'Segment'):
        l1 = self.start.distance(self._projection_point(other, typed='start'))
        l2 = self.end.distance(self._projection_point(other, typed='end'))
        return min(l1, l2)

    def angle_distance(self, other: 'Segment'):
        self_vector = self.end - self.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)

        if self_dist < self.eps:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist < self.eps:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())

        cos_theta = self_vector.dot(other.end - other.start) / (self.end.distance(self.start) * other.end.distance(other.start))
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length

    def _projection_point(self, other: 'Segment', typed="e"):
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        u = tmp.dot(self.end-self.start) / max(math.pow(self.end.distance(self.start), 2), 0.000001)
        return self.start + (self.end-self.start) * u

    def get_all_distance(self, seg: 'Segment'):
        res = self.angle_distance(seg)
        if str(self.start) != str(self.end):
            res += self.parallel_distance(seg)
        if self.traj_id != seg.traj_id:
            res += self.perpendicular_distance(seg)
        return res


def compare(segment_a: Segment, segment_b: Segment) -> Tuple[Segment, Segment]:
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)
