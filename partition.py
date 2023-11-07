import math
from segment import Segment
from point import _point2line_distance

eps = 1e-12   # defined the segment length theta, if length < eps then l_h=0


def segment_mdl_comp(traj, start_index, current_index, typed='par'):
    length_hypothesis = 0
    length_data_hypothesis_perpend = 0
    length_data_hypothesis_angle = 0

    seg = Segment(traj[start_index], traj[current_index])
    if typed == "par" or typed == "PAR":
        if seg.length < eps:
            length_hypothesis = 0
        else:
            length_hypothesis = math.log2(seg.length)

    # compute the segment hypothesis
    for i in range(start_index, current_index, 1):
        sub_seg = Segment(traj[i], traj[i+1])
        if typed == 'par' or typed == 'PAR':
            length_data_hypothesis_perpend += seg.perpendicular_distance(sub_seg)
            length_data_hypothesis_angle += seg.angle_distance(sub_seg)
        elif typed == "nopar" or typed == "NOPAR":
            length_hypothesis += sub_seg.length

    if typed == 'par' or typed == 'PAR':
        if length_data_hypothesis_perpend > eps:
            length_hypothesis += math.log2(length_data_hypothesis_perpend)
        if length_data_hypothesis_angle > eps:
            length_hypothesis += math.log2(length_data_hypothesis_angle)
        return length_hypothesis
    elif typed == "nopar" or typed == "NOPAR":
        if length_hypothesis < eps:
            return 0
        else:
            return math.log2(length_hypothesis)  # when typed == nopar the L(D|H) is zero.
    else:
        raise ValueError("The parameter 'typed' given value has error!")


def approximate_trajectory_partitioning(traj, traj_id=None, theta=5.0):
    size = len(traj)
    start_index: int = 0; length: int = 1

    partition_trajectory = []
    while (start_index + length) < size:
        curr_index = start_index + length
        cost_par = segment_mdl_comp(traj, start_index, curr_index, typed='par')
        cost_nopar = segment_mdl_comp(traj, start_index, curr_index, typed='nopar')
        if cost_par > (cost_nopar+theta):
            seg = Segment(traj[start_index], traj[curr_index-1], traj_id=traj_id)
            partition_trajectory.append(seg)
            start_index = curr_index - 1
            length = 1
        else:
            length += 1
    seg = Segment(traj[start_index], traj[size-1], traj_id=traj_id, cluster_id=-1)
    partition_trajectory.append(seg)
    return partition_trajectory


def rdp_trajectory_partitioning(trajectory, traj_id=None, epsilon=1.0):
    size = len(trajectory)
    d_max = 0.0
    index = 0
    for i in range(1, size-1, 1):
        d = _point2line_distance(trajectory[i].as_array(), trajectory[0].as_array(), trajectory[-1].as_array())
        if d > d_max:
            d_max = d
            index = i

    if d_max > epsilon:
        result = rdp_trajectory_partitioning(trajectory[:index+1], epsilon=epsilon) + \
                 rdp_trajectory_partitioning(trajectory[index:], epsilon=epsilon)
    else:
        result = [Segment(trajectory[0], trajectory[-1], traj_id=traj_id, cluster_id=-1)]
    return result
