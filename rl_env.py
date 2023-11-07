import numpy as np
import data_utils as F
from sortedcontainers import SortedList
import random

class TrajComp():
    def __init__(self, path, s, e, K, LEN, DB_DISTRI='', ID2Grids='', label='T'):
        self.K = K
        self.LEN = LEN
        self.pre = 0.0
        self.F_ward = {} # save (trajID, pointID)
        self._load(path, s, e)
        self.sortedlist = {}
        self.grid_has_tids = {}
        self.trajID2Grid = {}
        self.DB_DISTRI, self.ID2Grids, self.label = DB_DISTRI, ID2Grids, label
        self.action_container = []
        if DB_DISTRI != '' and label == 'T':
            self.Rtree_sim = F.Rtree()
            self.Rtree_sim_c = 0

    def _load(self, path, s, e):
        self.ori_traj_set = []
        self.DB_size = 0
        for num in range(s, e):
            traj = F.to_traj(path + str(num))
            self.ori_traj_set.append(traj)
            self.DB_size += len(traj)
    
    def id2pts(self, trajID, pointID):
        point = self.ori_traj_set[trajID][pointID]
        return (point[0],point[1],point[2],point[0],point[1],point[2])
    
    def read(self, s, e):
        self.F_ward[s] = e
        self.F_ward[e] = (None, None)
        if self.label == 'T':
            (s_trajID, s_pointID) = s
            (e_trajID, e_pointID) = e
            self.Rtree_sim.insert(self.Rtree_sim_c, self.id2pts(s_trajID, s_pointID), s_trajID)
            self.Rtree_sim_c += 1
            self.Rtree_sim.insert(self.Rtree_sim_c, self.id2pts(e_trajID, e_pointID), e_trajID)
            self.Rtree_sim_c += 1
    
    def read_m(self, s, m, e):
        self.F_ward[s] = m
        self.F_ward[m] = e
        if self.label == 'T':
            (m_trajID, m_pointID) = m
            self.Rtree_sim.insert(self.Rtree_sim_c, self.id2pts(m_trajID, m_pointID), m_trajID)
            self.Rtree_sim_c += 1
    
    def update_trajID2Grid(self, grid, trajID):
        if trajID in self.trajID2Grid:
            self.trajID2Grid[trajID].add(grid)
        else:
            self.trajID2Grid[trajID] = set([grid])
    
    def add_buffer(self, s, e):
        (trajID, s_pointID) = s
        (trajID, e_pointID) = e
        
        container = {}
        for m_pointID in range(s_pointID+1, e_pointID): 
            if (trajID, m_pointID) not in self.ID2Grids:
                continue
            seg = [self.ori_traj_set[trajID][s_pointID], self.ori_traj_set[trajID][m_pointID], self.ori_traj_set[trajID][e_pointID]]
            Eu, Et = F.Eu(seg), F.Et(seg)
            m = (trajID, m_pointID)
            grids = self.ID2Grids[m]
            tri = (s, m, e)
            state = (-Eu, Et)
            for grid in grids:
                if grid in container:
                    container[grid].add((state, tri))
                else:
                    container[grid] = SortedList({(state, tri)})
        
        for grid in container:
            val = container[grid][0]
            self.update_trajID2Grid(grid, trajID)
            if grid in self.sortedlist:
                if trajID not in self.grid_has_tids[grid]:
                    self.sortedlist[grid].add(val)
                    self.grid_has_tids[grid][trajID] = SortedList({val})
                else:
                    if len(self.grid_has_tids[grid][trajID]) == 0:                        
                        self.grid_has_tids[grid][trajID].add(val)
                        self.sortedlist[grid].add(val)
                    else:
                        val_ = self.grid_has_tids[grid][trajID][0]
                        if val < val_:
                            self.sortedlist[grid].remove(val_)
                            self.sortedlist[grid].add(val)
                        self.grid_has_tids[grid][trajID].add(val)
            else:
                self.sortedlist[grid] = SortedList({val})
                self.grid_has_tids[grid] = {}
                self.grid_has_tids[grid][trajID] = SortedList({val})
    
    def get_agent_inb_state(self, grid_s):
        while len(self.sortedlist[grid_s]) == 0:
            grid_s = random.sample(list(self.sortedlist.keys()), 1)[0]
        self.action_container = (self.sortedlist[grid_s])[:self.K] #
        tmp = []
        for (state, tri) in self.action_container:
            state = list(state)
            state[0] *= -1
            tmp.extend(state)
        if len(tmp) < self.K*self.LEN:
            times = int((self.K*self.LEN - len(tmp))/self.LEN)
            #print('times',times)
            tmp.extend(tmp[:self.LEN]*times)
            self.action_container.extend(self.action_container[:1]*times)
        return np.array(tmp).reshape(-1, self.K*self.LEN)                     
        
    def update_inb(self, action):
        (state, tri) = self.action_container[action]
        (s, m, e) = tri
        
        self.read_m(s, m, e)
        (trajID, s_pointID) = s
        (trajID, m_pointID) = m
        (trajID, e_pointID) = e
        
        for grid in self.trajID2Grid[trajID].copy():
            tmp = self.grid_has_tids[grid][trajID][0]
            if not (s_pointID < tmp[1][1][1] and tmp[1][1][1] < e_pointID):
                continue
            #print(tmp, self.sortedlist[grid])
            self.sortedlist[grid].remove(tmp)
            del self.grid_has_tids[grid][trajID][0]

            if len(self.grid_has_tids[grid][trajID]) > 0:
                self.sortedlist[grid].add(self.grid_has_tids[grid][trajID][0])
            if len(self.grid_has_tids[grid][trajID]) == 0:
                self.trajID2Grid[trajID].remove(grid)            

        self.add_buffer((trajID, s_pointID), (trajID, m_pointID))
        self.add_buffer((trajID, m_pointID), (trajID, e_pointID))
        
        for grid in self.trajID2Grid[trajID].copy():
            for (_state, _tri) in self.grid_has_tids[grid][trajID]:
                if _tri[0] == s and _tri[-1] == e:
                    self.grid_has_tids[grid][trajID].remove((_state, _tri))
                    if (_state, _tri) in self.sortedlist[grid]:
                        self.sortedlist[grid].remove((_state, _tri))
                        if len(self.grid_has_tids[grid][trajID]) > 0:
                            self.sortedlist[grid].add(self.grid_has_tids[grid][trajID][0])    

    def get_simplified_DB(self):
        sim_DB = []
        len_ = len(self.ori_traj_set)
        for i in range(0, len_):
            head = 0
            sim_tmp = []
            while (i, head) in self.F_ward:
                sim_tmp.append(self.ori_traj_set[i][head])
                head = self.F_ward[(i, head)][1]
            sim_DB.append(sim_tmp)
        return sim_DB
        
    def submit_query(self, Rtree_ref='', DB_QUERY=''):
        if self.label == 'T':
            F1 = F.range_query_operator(Rtree_ref, self.Rtree_sim, DB_QUERY, verbose=False)
            reward = F1 - self.pre
            self.pre = F1
            return reward
        else:
            sim_DB = self.get_simplified_DB()
            self.Rtree_sim, _ = F.build_Rtree(sim_DB)
            F1 = F.range_query_operator(Rtree_ref, self.Rtree_sim, DB_QUERY, verbose=False)
            return F1
    
class TrajSam():
    def __init__(self, traj_set, level_start, label='data'):
        np.random.seed(0)
        self.label = label
        self.ori_traj_set = traj_set
        self.DB_DISTRI_trajID = {}
        self.Grid2IDs = {}
        self.ID2Grids = {}
        self.ID2Obs = {}
        self.cubeID = 0
        self.Xmin, self.Ymin, self.Tmin, self.Xmax, self.Ymax, self.Tmax = 1.044024, -179.9695933, 1176341492, 21.524024, -159.48959330000002, 1795656692
        self.x_step, self.y_step, self.t_step = (self.Xmax-self.Xmin)/(2**level_start), (self.Ymax-self.Ymin)/(2**level_start), (self.Tmax-self.Tmin)/(2**level_start)
        if label == 'data':
            self.data_sampling()
        if label == 'gau':
            self.gau_sampling()
            
    def data_sampling(self):
        for trajID in range(len(self.ori_traj_set)):
            for pointID in range(len(self.ori_traj_set[trajID])):
                if pointID == 0 or pointID == len(self.ori_traj_set[trajID]) - 1:
                    continue
                [x, y, t] = self.ori_traj_set[trajID][pointID]
                key = tuple([int((x - self.Xmin)/self.x_step), int((y - self.Ymin)/self.y_step), int((t - self.Tmin)/self.t_step), self.x_step]) #x_step is to distinguish diff levels
                if key in self.DB_DISTRI_trajID:
                    self.DB_DISTRI_trajID[key].add(trajID)
                    self.Grid2IDs[key].append((trajID, pointID))
                else:
                    self.DB_DISTRI_trajID[key] = set([trajID])
                    self.Grid2IDs[key] = [(trajID, pointID)]
        DB_DISTRI = {}
        for key in self.DB_DISTRI_trajID:
            if len(self.DB_DISTRI_trajID[key]) > 1:
                DB_DISTRI[key] = len(self.DB_DISTRI_trajID[key])
        self.sample_key = list(DB_DISTRI.keys())
        self.sample_value = np.array(list(DB_DISTRI.values()))
        self.sample_value = self.sample_value/np.sum(self.sample_value)
    
    def data_get_observation(self):
        self.level_cur += 1
        self.key_container = []
        if self.cubeID in self.ID2Obs:
            self.key_container = self.ID2Obs[self.cubeID][1]
            return self.ID2Obs[self.cubeID][0]
        else:
            TP_IDs = self.Grid2IDs[self.cubeID]
            self.x_step, self.y_step, self.t_step = (self.Xmax-self.Xmin)/(2**self.level_cur), (self.Ymax-self.Ymin)/(2**self.level_cur), (self.Tmax-self.Tmin)/(2**self.level_cur)
            for (trajID, pointID) in TP_IDs:
                [x, y, t] = self.ori_traj_set[trajID][pointID]
                key = tuple([int((x - self.Xmin)/self.x_step), int((y - self.Ymin)/self.y_step), int((t - self.Tmin)/self.t_step), self.x_step])
                self.key_container.append(key)
                if key in self.DB_DISTRI_trajID:
                    self.DB_DISTRI_trajID[key].add(trajID)
                    self.Grid2IDs[key].append((trajID, pointID))
                else:
                    self.DB_DISTRI_trajID[key] = set([trajID])
                    self.Grid2IDs[key] = [(trajID, pointID)]
            self.key_container = list(set(self.key_container))
            #print('number of childs', len(self.key_container), 'in max 8')
            tmp, total, padv, padk = [], 0, 0, 0
            for key in self.key_container:
                total += len(self.DB_DISTRI_trajID[key])
                if len(self.DB_DISTRI_trajID[key]) > padv:
                    padv = len(self.DB_DISTRI_trajID[key])
                    padk = key
            for key in self.key_container:
                tmp.append(len(self.DB_DISTRI_trajID[key])/total)#for data
                tmp.append(len(self.DB_DISTRI_trajID[key])/total)#for query
            while len(tmp) < 2*8: #8 childs and each for 2 observations
                tmp.append(padv/total)
                tmp.append(padv/total)
                self.key_container.append(padk)            
            Obs = np.array(tmp).reshape(-1, 16)
            self.ID2Obs[self.cubeID] = [Obs, self.key_container]
            return Obs

    def gau_sampling(self):
        X, Y, T, DB_DISTRI = [], [], [], {}
        self.gau_workload = {}
        for trajID in range(len(self.ori_traj_set)):
            for pointID in range(len(self.ori_traj_set[trajID])):
                if pointID == 0 or pointID == len(self.ori_traj_set[trajID]) - 1:
                    continue
                point = self.ori_traj_set[trajID][pointID]
                [x, y, t] = point
                key = tuple([int((x - self.Xmin)/self.x_step), int((y - self.Ymin)/self.y_step), int((t - self.Tmin)/self.t_step), self.x_step])
                if key in self.DB_DISTRI_trajID:
                    self.DB_DISTRI_trajID[key].add(trajID)
                    self.Grid2IDs[key].append((trajID, pointID))
                else:
                    self.DB_DISTRI_trajID[key] = set([trajID])
                    self.Grid2IDs[key] = [(trajID, pointID)]
                    X.append(key[0])
                    Y.append(key[1])
                    T.append(key[2])
        X.sort()
        Y.sort()
        T.sort()
        X_map, Y_map, T_map = {}, {}, {}
        for i in range(len(self.Grid2IDs)):
            X_map[i] = X[i]
            Y_map[i] = Y[i]
            T_map[i] = T[i]
        mu, alpha = (1+len(self.Grid2IDs))/2, (len(self.Grid2IDs)-1)/4
        count = 0
        while True:
            if count == 10000:
                break
            [x, y, t] = [np.random.normal(loc=mu, scale=alpha, size=None),
                         np.random.normal(loc=mu, scale=alpha, size=None),
                         np.random.normal(loc=mu, scale=alpha, size=None)]
            if (int(x) in X_map) and (int(y) in Y_map) and (int(t) in T_map):
                key = tuple([X_map[int(x)], Y_map[int(y)], T_map[int(t)], self.x_step])
                if key in self.Grid2IDs:
                    (trajID, pointID) = random.sample(self.Grid2IDs[key], 1)[0]
                    count += 1
                    if key in DB_DISTRI:
                        DB_DISTRI[key] += 1
                        self.gau_workload[key].append((trajID, pointID))
                    else:
                        DB_DISTRI[key] = 1
                        self.gau_workload[key] = [(trajID, pointID)]

        self.sample_key = list(DB_DISTRI.keys())
        self.sample_value = np.array(list(DB_DISTRI.values()))
        self.sample_value = self.sample_value/np.sum(self.sample_value)
        
    def gau_get_observation(self):
        self.level_cur += 1
        self.key_container = []
        if self.cubeID in self.ID2Obs:
            self.key_container = self.ID2Obs[self.cubeID][1]
            return self.ID2Obs[self.cubeID][0]
        else:
            TP_IDs = self.Grid2IDs[self.cubeID]
            self.x_step, self.y_step, self.t_step = (self.Xmax-self.Xmin)/(2**self.level_cur), (self.Ymax-self.Ymin)/(2**self.level_cur), (self.Tmax-self.Tmin)/(2**self.level_cur)
            for (trajID, pointID) in TP_IDs:
                [x, y, t] = self.ori_traj_set[trajID][pointID]
                key = tuple([int((x - self.Xmin)/self.x_step), int((y - self.Ymin)/self.y_step), int((t - self.Tmin)/self.t_step), self.x_step])
                self.key_container.append(key)
                if key in self.DB_DISTRI_trajID:
                    self.DB_DISTRI_trajID[key].add(trajID)
                    self.Grid2IDs[key].append((trajID, pointID))
                else:
                    self.DB_DISTRI_trajID[key] = set([trajID])
                    self.Grid2IDs[key] = [(trajID, pointID)]
            self.key_container = list(set(self.key_container))
            #print('number of childs', len(self.key_container), 'in max 8')
            
            if self.cubeID in self.gau_workload:
                GAU_TP_IDs = self.gau_workload[self.cubeID]
                for (trajID, pointID) in GAU_TP_IDs:
                    [x, y, t] = self.ori_traj_set[trajID][pointID]
                    key = tuple([int((x - self.Xmin)/self.x_step), int((y - self.Ymin)/self.y_step), int((t - self.Tmin)/self.t_step), self.x_step])
                    if key in self.gau_workload:
                        self.gau_workload[key].append((trajID, pointID))
                    else:
                        self.gau_workload[key] = [(trajID, pointID)]    
                tmp, total_d, total_q, padk, padv_d, padv_q = [], 0, 0, 0, 0, 0
                for key in self.key_container:
                    total_d += len(self.DB_DISTRI_trajID[key])
                    if key in self.gau_workload:
                        total_q += len(self.gau_workload[key])
                        if len(self.gau_workload[key]) > padv_q:
                            padv_q = len(self.gau_workload[key])
                            padv_d = len(self.DB_DISTRI_trajID[key])
                            padk = key
                
                for key in self.key_container:
                    tmp.append(len(self.DB_DISTRI_trajID[key])/total_d)#for data
                    if key in self.gau_workload:
                        tmp.append(len(self.gau_workload[key])/total_q)#for query
                    else:
                        tmp.append(0.0)
                while len(tmp) < 2*8: #8 childs and each for 2 observations
                    tmp.append(padv_d/total_d)
                    tmp.append(padv_q/total_q)
                    self.key_container.append(padk)            
            else:
                tmp, total, padv, padk = [], 0, 0, 0
                for key in self.key_container:
                    total += len(self.DB_DISTRI_trajID[key])
                    if len(self.DB_DISTRI_trajID[key]) > padv:
                        padv = len(self.DB_DISTRI_trajID[key])
                        padk = key
                for key in self.key_container:
                    tmp.append(len(self.DB_DISTRI_trajID[key])/total)#for data
                    tmp.append(0)#for query
                while len(tmp) < 2*8: #8 childs and each for 2 observations
                    tmp.append(padv/total)
                    tmp.append(0)
                    self.key_container.append(padk)            
            Obs = np.array(tmp).reshape(-1, 16)
            self.ID2Obs[self.cubeID] = [Obs, self.key_container]
            return Obs
    
    def starting(self, level_start):
        self.level_cur = level_start
        if self.label == 'data':
            index = int(np.random.choice(len(self.sample_value), 1, p=self.sample_value))
            self.cubeID = self.sample_key[index]
            return self.data_get_observation()
        if self.label == 'gau':
            index = int(np.random.choice(len(self.sample_value), 1, p=self.sample_value))
            self.cubeID = self.sample_key[index]
            return self.gau_get_observation()        
            
    def get_agent_oub_state(self, action):
        self.cubeID = self.key_container[action]
        if self.label == 'data':
            return self.data_get_observation()
        if self.label == 'gau':
            return self.gau_get_observation()
    
    def get_ID2Grids(self, SampleList_tra):
        tmp = set(SampleList_tra)
        for tar_grid in tmp:
            pts = self.Grid2IDs[tar_grid]
            for pt in pts:
                if pt not in self.ID2Grids:
                    self.ID2Grids[pt] = set([tar_grid])
                else:
                    self.ID2Grids[pt].add(tar_grid)
    