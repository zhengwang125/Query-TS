import data_utils as F
import pickle
import numpy as np

if __name__ == '__main__':
    traj_path = './TrajData/Geolife_out/'
    val_start = 7000
    traj_amount = 1000
    DB = []
    numpts = 0
    for i in range(val_start, val_start+traj_amount):
        traj = F.to_traj(traj_path + str(i))
        numpts += len(traj)
        DB.append(traj)
    print('#pts', numpts)
    DB_TREE, _ = F.build_Rtree(DB)
    
    
    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_data(DB)
    gen_distri, query1, query2 = F.get_query_workload_data(DB_DISTRI)
    for ratio in [0.0025]:
        name = 'data_queryts_'+str(val_start)+'_'+str(val_start+traj_amount)+'_'+str(ratio)+'_geolife'
        sim_DB = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        simDB_TREE, _  = F.build_Rtree(sim_DB)
        RES = F.join_query_operator(DB, sim_DB, DB_TREE, simDB_TREE, query1+query2)
        print('join(similarity) query effectiveness (data distribution)', RES)
    
    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_gau(DB)    
    gen_distri, query1, query2 = F.get_query_workload_gau(DB_DISTRI)
    for ratio in [0.0025]:
        name = 'gau_queryts_'+str(val_start)+'_'+str(val_start+traj_amount)+'_'+str(ratio)+'_geolife'
        sim_DB = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        simDB_TREE, _  = F.build_Rtree(sim_DB)
        RES = F.join_query_operator(DB, sim_DB, DB_TREE, simDB_TREE, query1+query2)
        print('join(similarity) query effectiveness (gau distribution)', RES)
        
