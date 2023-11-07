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
    clu_name = 'clustering_workload'
    gen_distri, query1, query2 = F.get_query_workload_data(DB_DISTRI)
    traj_clus = F.clustering_offline(DB, DB_TREE, query1+query2)
    pickle.dump(traj_clus, open(traj_path+clu_name, 'wb'), protocol=2)
    traj_clus = pickle.load(open(traj_path+clu_name, 'rb'), encoding='bytes')
    for ratio in [0.0025]:
        name = 'data_queryts_'+str(val_start)+'_'+str(val_start+traj_amount)+'_'+str(ratio)+'_geolife'
        sim_DB = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        simDB_TREE, _  = F.build_Rtree(sim_DB)
        RES = F.clustering_online(traj_clus, sim_DB, simDB_TREE, query1+query2)
        print('clustering effectiveness (data distribution)', RES)
  
    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_gau(DB)    
    clu_name_gau = 'clustering_workload_gau'
    gen_distri, query1, query2 = F.get_query_workload_gau(DB_DISTRI)
    traj_clus = F.clustering_offline(DB, DB_TREE, query1+query2)
    pickle.dump(traj_clus, open(traj_path+clu_name_gau, 'wb'), protocol=2)
    traj_clus = pickle.load(open(traj_path+clu_name_gau, 'rb'), encoding='bytes')
    for ratio in [0.0025]:
        name = 'gau_queryts_'+str(val_start)+'_'+str(val_start+traj_amount)+'_'+str(ratio)+'_geolife'
        sim_DB = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        simDB_TREE, _  = F.build_Rtree(sim_DB)
        RES = F.clustering_online(traj_clus, sim_DB, simDB_TREE, query1+query2)
        print('clustering effectiveness (gau distribution)', RES)
