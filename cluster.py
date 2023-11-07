import math
from segment import compare, Segment
from point import Point
from collections import deque, defaultdict

min_traj_cluster = 2

def neighborhood(seg, segs, epsilon=2.0):
    segment_set = []
    for segment_tmp in segs:
        seg_long, seg_short = compare(seg, segment_tmp)  # get long segment by compare segment
        if seg_long.get_all_distance(seg_short) <= epsilon:
            segment_set.append(segment_tmp)
    return segment_set


def expand_cluster(segs, queue: deque, cluster_id: int, epsilon: float, min_lines: int):
    while len(queue) != 0:
        curr_seg = queue.popleft()
        curr_num_neighborhood = neighborhood(curr_seg, segs, epsilon=epsilon)
        if len(curr_num_neighborhood) >= min_lines:
            for m in curr_num_neighborhood:
                if m.cluster_id == -1:
                    queue.append(m)
                    m.cluster_id = cluster_id
        else:
            pass

def line_segment_clustering(traj_segments, epsilon: float = 2.0, min_lines: int = 5):
    cluster_id = 0
    cluster_dict = defaultdict(list)
    for seg in traj_segments:
        _queue = deque(list(), maxlen=50)
        if seg.cluster_id == -1:
            seg_num_neighbor_set = neighborhood(seg, traj_segments, epsilon=epsilon)
            if len(seg_num_neighbor_set) >= min_lines:
                seg.cluster_id = cluster_id
                for sub_seg in seg_num_neighbor_set:
                    sub_seg.cluster_id = cluster_id  # assign clusterId to segment in neighborhood(seg)
                    _queue.append(sub_seg)  # insert sub segment into queue
                expand_cluster(traj_segments, _queue, cluster_id, epsilon, min_lines)
                cluster_id += 1
            else:
                seg.cluster_id = -1
        # print(seg.cluster_id, seg.traj_id)
        if seg.cluster_id != -1:
            cluster_dict[seg.cluster_id].append(seg)

    remove_cluster = dict()
    cluster_number = len(cluster_dict)
    for i in range(0, cluster_number):
        traj_num = len(set(map(lambda s: s.traj_id, cluster_dict[i])))
        #print("the %d cluster lines:" % i, traj_num)
        if traj_num < min_traj_cluster:
            remove_cluster[i] = cluster_dict.pop(i)
    return cluster_dict, remove_cluster


def representative_trajectory_generation(cluster_segment: dict, min_lines: int = 3, min_dist: float = 2.0):
    representive_point = defaultdict(list)
    for i in cluster_segment.keys():
        cluster_size = len(cluster_segment.get(i))
        sort_point = []  # [Point, ...], size = cluster_size*2
        rep_point, zero_point = Point(0, 0, -1), Point(1, 0, -1)

        for j in range(cluster_size):
            rep_point = rep_point + (cluster_segment[i][j].end - cluster_segment[i][j].start)
        rep_point = rep_point / float(cluster_size)

        cos_theta = rep_point.dot(zero_point) / rep_point.distance(Point(0, 0, -1))  # cos(theta)
        sin_theta = math.sqrt(1 - math.pow(cos_theta, 2))  # sin(theta)

        for j in range(cluster_size):
            s, e = cluster_segment[i][j].start, cluster_segment[i][j].end
            cluster_segment[i][j] = Segment(Point(s.x * cos_theta + s.y * sin_theta, s.y * cos_theta - s.x * sin_theta, -1),
                                            Point(e.x * cos_theta + e.y * sin_theta, e.y * cos_theta - e.x * sin_theta, -1),
                                            traj_id=cluster_segment[i][j].traj_id,
                                            cluster_id=cluster_segment[i][j].cluster_id)
            sort_point.extend([cluster_segment[i][j].start, cluster_segment[i][j].end])

        sort_point = sorted(sort_point, key=lambda _p: _p.x)
        for p in range(len(sort_point)):
            intersect_cnt = 0.0
            start_y = Point(0, 0, -1)
            for q in range(cluster_size):
                s, e = cluster_segment[i][q].start, cluster_segment[i][q].end
                if (sort_point[p].x <= e.x) and (sort_point[p].x >= s.x):
                    if s.x == e.x:
                        continue
                    elif s.y == e.y:
                        intersect_cnt += 1
                        start_y = start_y + Point(sort_point[p].x, s.y, -1)
                    else:
                        intersect_cnt += 1
                        start_y = start_y + Point(sort_point[p].x, (e.y-s.y)/(e.x-s.x)*(sort_point[p].x-s.x)+s.y, -1)
            if intersect_cnt >= min_lines:
                tmp_point: Point = start_y / intersect_cnt
                tmp = Point(tmp_point.x*cos_theta-sin_theta*tmp_point.y,
                            sin_theta*tmp_point.x+cos_theta*tmp_point.y, -1)
                _size = len(representive_point[i]) - 1
                if _size < 0 or (_size >= 0 and tmp.distance(representive_point[i][_size]) > min_dist):
                    representive_point[i].append(tmp)
    return representive_point
