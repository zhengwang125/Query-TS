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
    t2v_name_data = 'knn_query_win_t2v_workload_data'
    gen_distri, query1, query2 = F.get_query_workload_data(DB_DISTRI)
    GroundQuerySet, interval = F.knn_t2v_query_offline(DB, DB_TREE, query1+query2)
    pickle.dump([GroundQuerySet, interval], open(traj_path+t2v_name_data, 'wb'), protocol=2)
    [GroundQuerySet, interval] = pickle.load(open(traj_path+t2v_name_data, 'rb'), encoding='bytes')
    for ratio in [0.0025]:
        name = 'data_queryts_'+str(val_start)+'_'+str(val_start+traj_amount)+'_'+str(ratio)+'_geolife'
        sim_DB = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        simDB_TREE, _  = F.build_Rtree(sim_DB)
        RES = F.knn_t2v_query_online(GroundQuerySet, interval, simDB_TREE, sim_DB)
        print('knn t2v query effectiveness (data distribution)', RES)

    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_gau(DB)  
    t2v_name_gau = 'knn_query_win_t2v_workload_gau'
    gen_distri, query1, query2 = F.get_query_workload_gau(DB_DISTRI)
    GroundQuerySet, interval = F.knn_t2v_query_offline(DB, DB_TREE, query1+query2)
    pickle.dump([GroundQuerySet, interval], open(traj_path+t2v_name_gau, 'wb'), protocol=2)
    [GroundQuerySet, interval] = pickle.load(open(traj_path+t2v_name_gau, 'rb'), encoding='bytes')
    for ratio in [0.0025]:
        name = 'gau_queryts_'+str(val_start)+'_'+str(val_start+traj_amount)+'_'+str(ratio)+'_geolife'
        sim_DB = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        simDB_TREE, _  = F.build_Rtree(sim_DB)
        RES = F.knn_t2v_query_online(GroundQuerySet, interval, simDB_TREE, sim_DB)
        print('knn t2v query effectiveness (gau distribution)', RES)
