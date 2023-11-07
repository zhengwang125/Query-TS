from julia import Main
import os
Main.include('call.jl')
from t2vec_utils import args, model_init, submit
args.checkpoint = "./TrajData/best_model.pt"
args.vocab_size = 40004 #40004 (geolife)
m0 = model_init(args)
import numpy as np
import math
from rtree import index
import random
import shutil
from partition import approximate_trajectory_partitioning
from point import Point
from cluster import line_segment_clustering

print('done import')

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def points2meter(points):
    rtn = []
    for p in points:
        lon_meter, lat_meter = lonlat2meters(lon=p[1], lat=p[0])
        rtn.append([lat_meter,lon_meter,p[2]])
    return rtn

def to_traj(file):
    traj = []
    f = open(file)
    for line in f:
        temp = line.strip().split(' ')
        if len(temp) < 3:
            continue
        traj.append([float(temp[0]), float(temp[1]), int(float(temp[2]))])
    f.close()
    return traj

def Eu(segment):
    ps = segment[0]
    pe = segment[-1]    
    syn_time = segment[1][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    e = np.linalg.norm(np.array([segment[1][0],segment[1][1]]) - np.array([syn_x,syn_y]))
    return e

def Et(segment):
    ps = segment[0]
    pm = segment[1]
    pe = segment[-1]
    A = pe[1] - ps[1]
    B = ps[0] - pe[0]
    C = pe[0] * ps[1] - ps[0] * pe[1]
    if A == 0 and B == 0:
        return 0.0
    else:
        x = (B*B*pm[0] - A*B*pm[1] - A*C)/(A*A + B*B)
        y = (-A*B*pm[0] + A*A*pm[1] - B*C)/(A*A + B*B)
        speed = np.linalg.norm(np.array([ps[0], ps[1]]) - np.array([pe[0], pe[1]]))/(pe[2]-ps[2])
        return abs(ps[2] + np.linalg.norm(np.array([ps[0], ps[1]]) - np.array([x,y]))/speed - pm[2])

def sed_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            syn_time = segment[i][2]
            time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
            syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
            syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
            e = max(e, np.linalg.norm(np.array([segment[i][0],segment[i][1]]) - np.array([syn_x,syn_y])))
        return e
    
def sed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, sed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def ped_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            pm = segment[i]
            A = pe[1] - ps[1]
            B = ps[0] - pe[0]
            C = pe[0] * ps[1] - ps[0] * pe[1]
            if A == 0 and B == 0:
                e = max(e, 0.0)
            else:
                e = max(e, abs((A * pm[0] + B * pm[1] + C)/ np.sqrt(A * A + B * B)))
        return e

def ped_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, ped_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def angle(v1):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    angle1 = math.atan2(dy1, dx1)
    if angle1 >= 0:
        return angle1
    else:
        return 2*math.pi + angle1
    
def dad_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        theta_0 = angle([ps[0],ps[1],pe[0],pe[1]])
        for i in range(0,len(segment)-1):
            pm_0 = segment[i]
            pm_1 = segment[i+1]
            theta_1 = angle([pm_0[0],pm_0[1],pm_1[0],pm_1[1]])
            e = max(e, min(abs(theta_0 - theta_1), 2*math.pi - abs(theta_0 - theta_1)))
        return e

def dad_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, dad_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def get_point(ps, pe, segment, index):
    syn_time = segment[index][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    return [syn_x, syn_y], syn_time

def speed_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(0,len(segment)-1):
            p_1, t_1 = get_point(ps, pe, segment, i)
            p_2, t_2 = get_point(ps, pe, segment, i+1)
            time = 1 if t_2 - t_1 == 0 else abs(t_2-t_1)
            est_speed = np.linalg.norm(np.array(p_1) - np.array(p_2))/time
            rea_speed = np.linalg.norm(np.array([segment[i][0], segment[i][1]]) - np.array([segment[i+1][0], segment[i+1][1]]))/time
            e = max(e, abs(est_speed - rea_speed))
        return e

def speed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, speed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

class Rtree():
    def __init__(self, *args):
        self.p = index.Property()
        self.p.dimension = 3
        if len(args) == 0:
            self.idx = index.Index(properties=self.p)
        else:
            self.idx = index.Index(args[0], properties=self.p)
    
    def insert(self, id, data, obj): #id = int, data = (lat, lon), obj_trajID
        self.idx.insert(id, data, obj=obj)
    
    def delete(self, id, data): #id = int, data = (lat, lon), obj_trajID
        self.idx.delete(id, data)

    def knn(self, width, num=1, objects=True):# width = (xmin, ymin, tmin, xmax, ymax, tmax)
        res=list(self.idx.nearest(width, num, objects=objects))
        return res
    
    def range_query(self, width, objects=True): #  width = (xmin, ymin, tmin, xmax, ymax, tmax)
        res=list(self.idx.intersection(width, objects=objects))
        return res
    
def save(filename1, filename2):
    shutil.copyfile(filename1+'.idx', filename2+'.idx')
    shutil.copyfile(filename1+'.dat', filename2+'.dat')        

def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def files_clean(file, name):
    files=os.listdir(file)
    for i , f in enumerate(files):
        if f.find(name)>=0:
            os.remove(file+f)
            
def build_Rtree(DB, filename=''): # can set a save=''
    #query on ref_DB
    if filename=='':
        Rtree_ = Rtree()
    else:
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx')
        Rtree_ = Rtree(filename)
    c = 0
    delete_rec = {}
    for trajID in range(len(DB)):
        for pointID in range(len(DB[trajID])):
            point = DB[trajID][pointID]
            Rtree_.insert(c, (point[0],point[1],point[2],point[0],point[1],point[2]), trajID)
            delete_rec[(trajID, pointID)] = c
            c += 1
    return Rtree_, delete_rec
    
def build_Rtree_Each(DB, file='', name=''):
    #query on ref_DB
    files_clean(file, name)
    print('finished clean')
    filename = file+name
    c = 0
    for trajID in range(len(DB),-1,-1):
        if trajID == len(DB):
            Rtree(filename+str(trajID)).idx.close()
            shutil.copyfile(filename+str(trajID)+'.idx',filename+str(trajID-1)+'.idx')
            shutil.copyfile(filename+str(trajID)+'.dat',filename+str(trajID-1)+'.dat')
            continue
        Rtree_ = Rtree(filename+str(trajID))
        for pointID in range(len(DB[trajID])):
            point = DB[trajID][pointID]
            Rtree_.insert(c, (point[0],point[1],point[2],point[0],point[1],point[2]), trajID)
            c += 1
        Rtree_.idx.close()
        shutil.copyfile(filename+str(trajID)+'.idx',filename+str(trajID-1)+'.idx')
        shutil.copyfile(filename+str(trajID)+'.dat',filename+str(trajID-1)+'.dat')    

def enlarge_Rtree(obj, DB, from_):
    if obj == '':
        obj = Rtree()
    c = 0
    for trajID in range(len(DB)):
        for pointID in range(len(DB[trajID])):
            point = DB[trajID][pointID]
            obj.insert(c, (point[0],point[1],point[2],point[0],point[1],point[2]), trajID+from_)
            c += 1
    return obj

def obtain_Rtree(filename):
    Rtree_ = Rtree(filename)
    return Rtree_

def get_distribution_feature_data(db):
    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = {}, {}, {}
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    x_step, y_step, t_step, thre = 0.02, 0.02, 3600*24*7, 1#3600*24*7
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in DB_DISTRI_trajID:
                DB_DISTRI_trajID[key].add(trajID)
            else:
                DB_DISTRI_trajID[key] = set([trajID])
    for key in DB_DISTRI_trajID:
        if len(DB_DISTRI_trajID[key]) > thre:
            DB_DISTRI[key] = len(DB_DISTRI_trajID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID

def get_distribution_feature_gau(db):
    DB_DISTRI, ID2Grid, Grid2ID, DB_DISTRI_trajID = {}, {}, {}, {}
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    x_step, y_step, t_step = 0.02, 0.02, 3600*24*7
    X, Y, T = [], [], []
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in Grid2ID:
                Grid2ID[key].add(trajID)
            else:
                Grid2ID[key] = set([trajID])
                X.append(key[0])
                Y.append(key[1])
                T.append(key[2])
    X.sort()
    Y.sort()
    T.sort()
    X_map, Y_map, T_map = {}, {}, {}
    for i in range(len(Grid2ID)):
        X_map[i] = X[i]
        Y_map[i] = Y[i]
        T_map[i] = T[i]
    mu, alpha = (1+len(Grid2ID))/2, (len(Grid2ID)-1)/4
    count = 0
    while True:
        if count == 10000:
            break
        [x, y, t] = [np.random.normal(loc=mu, scale=alpha, size=None),
                     np.random.normal(loc=mu, scale=alpha, size=None),
                     np.random.normal(loc=mu, scale=alpha, size=None)]
        if (int(x) in X_map) and (int(y) in Y_map) and (int(t) in T_map):
            key = tuple([X_map[int(x)], Y_map[int(y)], T_map[int(t)]])
            if key in Grid2ID:
                count += 1
                if key in DB_DISTRI:
                    DB_DISTRI[key] += 1
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
                else:
                    DB_DISTRI[key] = 1
                    DB_DISTRI_trajID[key] = set()
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID

def get_distribution_feature_zipf(db, a):
    DB_DISTRI, ID2Grid, Grid2ID, DB_DISTRI_trajID = {}, {}, {}, {}
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    x_step, y_step, t_step = 0.02, 0.02, 3600*24*7
    X, Y, T = [], [], []
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in Grid2ID:
                Grid2ID[key].add(trajID)
            else:
                Grid2ID[key] = set([trajID])
                X.append(key[0])
                Y.append(key[1])
                T.append(key[2])
    X.sort()
    Y.sort()
    T.sort()
    X_map, Y_map, T_map = {}, {}, {}
    for i in range(len(Grid2ID)):
        X_map[i] = X[i]
        Y_map[i] = Y[i]
        T_map[i] = T[i]
    
    x_loc, y_loc, t_loc = 0.75, 0.25, 0.75
    xbase = (1+len(Grid2ID))*x_loc
    ybase = (1+len(Grid2ID))*y_loc
    tbase = (1+len(Grid2ID))*t_loc
    
    total = 10000
    count = 0
    xz, yz, tz = np.random.zipf(a=a,size=total), np.random.zipf(a=a,size=total), np.random.zipf(a=a,size=total)
    x_max, y_max, t_max = xz.max(), yz.max(), tz.max()
    x_scala, y_scala, t_scala = len(Grid2ID)/x_max/2, len(Grid2ID)/y_max/2, len(Grid2ID)/t_max/2
    
    for count in range(total):
        x = xbase+random.choice([1,-1])*random.uniform(x_scala*(xz[count]-1),x_scala*xz[count])
        y = ybase+random.choice([1,-1])*random.uniform(y_scala*(yz[count]-1),y_scala*yz[count])
        t = tbase+random.choice([1,-1])*random.uniform(t_scala*(tz[count]-1),t_scala*tz[count])
        if (int(x) in X_map) and (int(y) in Y_map) and (int(t) in T_map):
            key = tuple([X_map[int(x)], Y_map[int(y)], T_map[int(t)]])
            if key in Grid2ID:
                if key in DB_DISTRI:
                    DB_DISTRI[key] += 1
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
                else:
                    DB_DISTRI[key] = 1
                    DB_DISTRI_trajID[key] = set()
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID

def get_query_workload_data(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value/np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num/2)], query_workload[int(num/2):]

def get_query_workload_gau(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value/np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num/2)], query_workload[int(num/2):]

def get_query_workload_zipf(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value/np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num/2)], query_workload[int(num/2):]

def deform_SED_whole(DB, sim_DB):
    res = []
    for i in range(len(DB)):
        _, error = sed_error(DB[i], sim_DB[i])
        res.append(error)
    return sum(res)/len(res)*100*1000

def deform_SED_return(DB, sim_DB, Rtree_ref, Rtree_sim, QUERY):    
    #query on sim_DB
    res = []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    for i in range(len(QUERY)):
        db, sim_db = [], []
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        B = set([item.object for item in sim_R])
        if len(B) == 0:
            for ob in A:
                db.append(DB[ob])
                sim_db.append([DB[ob][0],DB[ob][-1]])
        else:
            for ob in B:
                db.append(DB[ob])
                sim_db.append(sim_DB[ob])
        res.append(deform_SED_whole(db, sim_db))
    return sum(res)/len(res)

class Sample():
    def __init__(self, db, x_step=0.02, y_step=0.02, t_step=3600*24*7):
        DB_DISTRI, ID2Grid, DB_DISTRI_trajID = {}, {}, {}
        Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
        for trajID in range(len(db)):
            for pointID in range(len(db[trajID])):
                if pointID == 0 or pointID == len(db[trajID]) - 1:
                    continue
                point = db[trajID][pointID]
                [x, y, t] = point
                key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
                ID2Grid[(trajID, pointID)] = key
                if key in DB_DISTRI_trajID:
                    DB_DISTRI_trajID[key].add(trajID)
                else:
                    DB_DISTRI_trajID[key] = set([trajID])
        for key in DB_DISTRI_trajID:
            if len(DB_DISTRI_trajID[key]) > 1:
                DB_DISTRI[key] = len(DB_DISTRI_trajID[key])
        
        self.sample_key = list(DB_DISTRI.keys())
        self.sample_value = np.array(list(DB_DISTRI.values()))
        self.sample_value = self.sample_value/np.sum(self.sample_value)
        self.DB_DISTRI = DB_DISTRI
        self.ID2Grid = ID2Grid
        
    def get_sample_list_data(self, num=1, random_state=''):
        self.SampleList = []
        if random_state != '':
            np.random.seed(random_state)
        while len(self.SampleList) < num:
            index = int(np.random.choice(len(self.sample_value), 1, p=self.sample_value))
            self.SampleList.append(self.sample_key[index])
        return self.SampleList
    
    def get_sample_list_uniform(self, num=1, random_state=''):
        self.SampleList = []
        if random_state != '':
            np.random.seed(random_state)
        while len(self.SampleList) < num:
            KK_ = random.sample(self.sample_key, 1)[0]
            self.SampleList.append(KK_)
        return self.SampleList

def range_query_operator(Rtree_ref, Rtree_sim, QUERY, verbose=False):    
    #query on sim_DB
    F1 = []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    ee, ef, fe, ff = 0, 0, 0, 0
    for i in range(len(QUERY)):
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        B = set([item.object for item in sim_R])
        
        if verbose:
            if A != B:
                print('A & B', A, B)
        if A == set() and B == set():
            ee += 1
            F1.append(1.0)
        if A == set() and B != set():
            ef += 1
            F1.append(0.0)
        if A != set() and B == set():
            fe += 1
            F1.append(0.0)
        if A != set() and B != set():
            ff += 1
            P = len(A&B)/len(B)
            R = len(A&B)/len(A)
            if (P+R) == 0:
                F1.append(0.0)
            else:
                F1.append((2*P*R)/(P+R))
    return sum(F1)/len(F1)

def edr(ts_a, ts_b, eps):
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))
    # Initialize the first row and column
    cost[0, 0] = 0
    for i in range(1, M):
        cost[i, 0] = i
    for j in range(1, N):
        cost[0, j] = j
    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(1, N):
            if np.linalg.norm(ts_a[i][0:2]-ts_b[j][0:2])<eps:
                choices = 0
            else:
                choices = 1
            cost[i, j] = min(cost[i-1, j-1]+choices, cost[i, j-1]+1, cost[i-1, j]+1)
    return cost[-1, -1]

def t2vec(ts_a, ts_b):
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    ts_a[:, [0, 1]], ts_b[:, [0, 1]] = ts_a[:, [1, 0]], ts_b[:, [1, 0]]
    ts_a, ts_b = ts_a[:,0:2], ts_b[:,0:2]
    ts_a_token, ts_b_token = Main.get_seq(ts_a.T), Main.get_seq(ts_b.T)
    _, each_step_forw_a = submit(m0, ts_a_token)
    _, each_step_forw_b = submit(m0, ts_b_token)
    return np.linalg.norm(each_step_forw_a[0,-1,:]-each_step_forw_b[0,-1,:])

def get_sync_traj(traj,start_time,end_time,A,B):
    D_ = []
    syn_time = start_time
    ps = traj[A]
    pe = traj[A+1]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    D_.append([syn_x,syn_y,syn_time])
    for idx in range(A+1, B+1):
        if syn_time==traj[idx][2]:
            continue
        else:
            D_.append(traj[idx])
    if D_[-1][2] != end_time:
        syn_time = end_time
        ps = traj[B]
        pe = traj[B+1]
        time_ratio = 1 if (pe[2] - ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
        syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
        syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
        D_.append([syn_x,syn_y,syn_time])
    return D_

def get_block_trajs(DB, A, xmin, ymin, tmin, xmax, ymax, tmax):
    ref_DB = []
    for a in A:
        traj = DB[a]
        ref_db = []
        for pts in traj:
            if pts[0]>=xmin and pts[0]<=xmax and pts[1]>=ymin and pts[1]<=ymax and pts[2]>=tmin and pts[2]<=tmax:
                ref_db.append(pts)
        ref_DB.append(ref_db)
    return ref_DB

def knn_edr_query_offline(DB, Rtree_ref, test_query):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    GroundQuerySet, interval, record = [], [], {}
    repeat = {}
    for i in range(len(test_query)):
        (x_idx, y_idx, t_idx) = test_query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        repeat[(x_idx, y_idx, t_idx)] = 1
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 1:
            interval.append((A, test_query[i]))
            ref_DB = get_block_trajs(DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            GroundSet, QuerySet = [], []
            for q_ in range(len(ref_DB)):
                query = ref_DB[q_]
                ground = []
                for c_ in range(len(ref_DB)):
                    data = ref_DB[c_]
                    if (A[q_], A[c_]) in record:
                        ground.append([record[(A[q_], A[c_])], A[c_]])
                    else:
                        tmp = edr(query, data, eps=0.02)
                        ground.append([tmp, A[c_]])
                        record[(A[q_], A[c_])] = tmp
                ground.sort(key=lambda s:(s[0]))
                GroundSet.append(ground)
                QuerySet.append(query)
            GroundQuerySet.append((GroundSet, QuerySet))
    return GroundQuerySet, interval

def knn_edr_query_online(GroundQuerySet, interval, Rtree_sim, sim_DB, k=3):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    result = []
    for (A, test_query), (GroundSet, QuerySet) in zip(interval, GroundQuerySet):
        (x_idx, y_idx, t_idx) = test_query
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if  len(set(A)) == 0 or len(set(B)) == 0:
            continue
        win_sim_DB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
        for ground, query in zip(GroundSet, QuerySet):
            predict = []
            for j in range(len(win_sim_DB)):
                predict.append([edr(query, win_sim_DB[j], eps=0.02), B[j]])         
            predict.sort(key=lambda s:(s[0]))              
            predict_tmp, ground_tmp = [], []
            for predict_i in range(0, min(k,len(predict))):
                predict_tmp.append(predict[predict_i][1])
            for ground_i in range(0, min(k,len(ground))):
                ground_tmp.append(ground[ground_i][1])
            result.append(len(set(predict_tmp)&set(ground_tmp))/min(k,len(ground)))
    return sum(result)/len(result)

def knn_t2v_query_offline(DB, Rtree_ref, test_query):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    GroundQuerySet, interval, record = [], [], {}
    repeat = {}
    for i in range(len(test_query)):
        (x_idx, y_idx, t_idx) = test_query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        repeat[(x_idx, y_idx, t_idx)] = 1
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 1:
            interval.append((A, test_query[i]))
            ref_DB = get_block_trajs(DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            GroundSet, QuerySet = [], []
            for q_ in range(len(ref_DB)):
                query = ref_DB[q_]
                ground = []
                for c_ in range(len(ref_DB)):
                    data = ref_DB[c_]
                    if (A[q_], A[c_]) in record:
                        ground.append([record[(A[q_], A[c_])], A[c_]])
                    else:
                        tmp = t2vec(query, data)
                        ground.append([tmp, A[c_]])
                        record[(A[q_], A[c_])] = tmp
                ground.sort(key=lambda s:(s[0]))
                GroundSet.append(ground)
                QuerySet.append(query)
            GroundQuerySet.append((GroundSet, QuerySet))
    return GroundQuerySet, interval

def knn_t2v_query_online(GroundQuerySet, interval, Rtree_sim, sim_DB, k=3):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    result = []
    for (A, test_query), (GroundSet, QuerySet) in zip(interval, GroundQuerySet):
        (x_idx, y_idx, t_idx) = test_query
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if  len(set(A)) == 0 or len(set(B)) == 0:
            continue        
        win_sim_DB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
        for ground, query in zip(GroundSet, QuerySet):
            predict = []
            for j in range(len(win_sim_DB)):
                predict.append([t2vec(query, win_sim_DB[j]), B[j]])         
            predict.sort(key=lambda s:(s[0]))  
            predict_tmp, ground_tmp = [], []
            for predict_i in range(0, min(k,len(predict))):
                predict_tmp.append(predict[predict_i][1])
            for ground_i in range(0, min(k,len(ground))):
                ground_tmp.append(ground[ground_i][1])
            result.append(len(set(predict_tmp)&set(ground_tmp))/min(k,len(ground)))            
    return sum(result)/len(result)

def join(Q_sync,D_sync,Q_start,Q_end,eps=0.01):
    for i in range(Q_start, Q_end):
        if np.linalg.norm(np.array(Q_sync[i])-np.array(D_sync[i]))<eps:
            continue
        else:
            return False
    return True

def sync(traj):
    dict_sync = {}
    for i in range(len(traj)-1):
        ps = traj[i]
        pe = traj[i+1]
        if pe[2] - ps[2] <= 1:
            dict_sync[ps[2]] = ps[0:2]
            dict_sync[pe[2]] = pe[0:2]
            continue
        else:
            dict_sync[ps[2]] = ps[0:2]
            for i in range(ps[2]+1,pe[2]):
                syn_time = i
                time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
                syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
                syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
                dict_sync[i] = [syn_x,syn_y]
            dict_sync[pe[2]] = pe[0:2]
    return dict_sync

def join_query_operator(ref_DB, sim_DB, Rtree_ref, Rtree_sim, query):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    F1 = []
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])        

        A = list(A)
        B = list(B)
        win_ref_DB = get_block_trajs(ref_DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
        win_sim_DB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)

        for q_ in range(len(win_ref_DB)):
            ground, predict = set(), set()
            Query = win_ref_DB[q_]
            Q_sync = sync(Query)
            Q_start, Q_end = Query[0][2], Query[-1][2]
            for c_ in range(len(win_ref_DB)):
                #for ground
                D1_sync = sync(win_ref_DB[c_])
                D_start, D_end = win_ref_DB[c_][0][2], win_ref_DB[c_][-1][2]
                #if Q_start>= D_start and Q_end<= D_end:
                if (D_start >= Q_start and D_start <= Q_end) or (D_end >= Q_start and D_end <= Q_end):
                    if join(Q_sync,D1_sync,max(Q_start, D_start),min(Q_end,D_end),eps=0.05):
                        ground.add(A[c_])
            for c_ in range(len(win_sim_DB)):
                #for predict
                D2_sync = sync(win_sim_DB[c_])
                D_start, D_end = win_sim_DB[c_][0][2], win_sim_DB[c_][-1][2]
                #if Q_start>= D_start and Q_end<= D_end:
                if (D_start >= Q_start and D_start <= Q_end) or (D_end >= Q_start and D_end <= Q_end):
                    if join(Q_sync,D2_sync,max(Q_start, D_start),min(Q_end,D_end),eps=0.05):
                        predict.add(B[c_])
                    
            if ground == set() and predict == set():
                F1.append(1.0)
            if ground == set() and predict != set():
                F1.append(0.0)
            if ground != set() and predict == set():
                F1.append(0.0)
            if ground != set() and predict != set():
                P = len(ground&predict)/len(predict)
                R = len(ground&predict)/len(ground)
                if (P+R) == 0:
                    F1.append(0.0)
                else:
                    F1.append((2*P*R)/(P+R))
    return sum(F1)/len(F1)

def call_traclus(trajs, A):
    traj_set = []
    for ts in trajs:
        traj_set.append([Point(ts[i:i+2][0], ts[i:i+2][1]) for i in range(0, len(ts), 2)])

    # part 1: partition
    all_segs = approximate_trajectory_partitioning(traj_set[0], theta=5.0, traj_id=A[0])
    for i in range(1, len(traj_set)):
         part = approximate_trajectory_partitioning(traj_set[i], theta=5.0, traj_id=A[i])
         all_segs += part
    
    norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=3, epsilon=0.03)

    return norm_cluster

def get_clusters(norm_cluster):
    clusters = []
    traj_cluster_dict = {}
    for nc in range(len(norm_cluster)):
        cluster=[]
        for segment in norm_cluster[nc]:
            cluster.append(segment.traj_id)
            if segment.traj_id in traj_cluster_dict:
                traj_cluster_dict[segment.traj_id].add(nc)
            else:
                traj_cluster_dict[segment.traj_id] = set()
                traj_cluster_dict[segment.traj_id].add(nc)            
        clusters.append(set(cluster))
    return clusters, traj_cluster_dict

def get_input(traj_db):
    ts=[]
    for traj in traj_db:
        ts.append(np.array(traj)[:,0:2].reshape(1,-1).tolist()[0])
    return ts

def clustering_offline(DB, DB_TREE, query):
    Ts_DB, ID = [], []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    repeat = {}
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = DB_TREE.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 0:
            ref_DB = get_block_trajs(DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            ts_DB = get_input(ref_DB)
            Ts_DB += ts_DB
            ID += A
    norm_cluster_DB = call_traclus(Ts_DB, ID)
    clusters_DB, traj_cluster_dict_DB = get_clusters(norm_cluster_DB)
    return traj_cluster_dict_DB
    
def clustering_online(traj_cluster_dict_DB, sim_DB, SIMDB_TREE, query):
    F1ALL, Ts_SIMDB, ID = [], [], []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    repeat = {}
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        sim_R = SIMDB_TREE.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if len(B) > 0:
            simDB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            ts_SIMDB = get_input(simDB)
            Ts_SIMDB += ts_SIMDB
            ID += B
    
    norm_cluster_simDB = call_traclus(Ts_SIMDB, ID)
    clusters_simDB, traj_cluster_dict_simDB = get_clusters(norm_cluster_simDB)
    
    CO, CS, COS = 0, 0, 0
    for i in range(len(sim_DB)-1):
        for j in range(i+1, len(sim_DB)):
            refind, simind = 0, 0
            if (i in traj_cluster_dict_DB) and (j in traj_cluster_dict_DB):
                if len(traj_cluster_dict_DB[i]&traj_cluster_dict_DB[j])!=0:
                    refind=1
            if (i in traj_cluster_dict_simDB) and (j in traj_cluster_dict_simDB):
                if len(traj_cluster_dict_simDB[i]&traj_cluster_dict_simDB[j])!=0:
                    simind=1
            CO+=refind
            CS+=simind
            if refind==1 and simind==1:
                COS+=1
    if CS == 0 or CO == 0:
        return 0
    P = COS/CS
    R = COS/CO
    F1 = (2*P*R)/(P+R)
    F1ALL.append(F1)
    return sum(F1ALL)/len(F1ALL)

def get_xyt_min(ref_DB):
    X = []
    Y = []
    T = []
    c = 0
    for trajID in range(len(ref_DB)):
        for pointID in range(len(ref_DB[trajID])):
            point = ref_DB[trajID][pointID]
            X.append(point[0])
            Y.append(point[1])
            T.append(point[2])
            c += 1
    xmin, ymin, tmin, xmax, ymax, tmax = min(X), min(Y), min(T), max(X), max(Y), max(T)
    
if __name__ == '__main__':
    print('This is the data util.')
