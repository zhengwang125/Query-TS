# -*- utf-8 -*-
from rl_env import TrajComp, TrajSam
import time
from rl_brain import AGENT_Sel, AGENT_Sam
import warnings
import data_utils as F
import random
random.seed(0)
warnings.filterwarnings("ignore")

globaltime = str(time.time())

def run_dqn():
    SampleList_val = []
    for sam_i in range(int(sam_size_v)):
        observation_oub = env_rl_sam_v.starting(level_start)
        while True:
            action_oub = RL_OUB.online_act(observation_oub)
            if action_oub == 8 or env_rl_sam_v.level_cur == level_end:
                break
            observation_oub = env_rl_sam_v.get_agent_oub_state(action_oub)
        SampleList_val.append(env_rl_sam_v.cubeID)
    print('done sampling')
    env_rl_sam_v.get_ID2Grids(SampleList_val)
    print('init the whole database for training with sample list size', len(SampleList_val))
    
    env_rl_sel_v = TrajComp(traj_path, val_start, val_start + validDB_amount, K, LEN, val_DB_DISTRI, env_rl_sam_v.ID2Grids, 'V')
    buffer_valid_size = int(ratio*env_rl_sel_v.DB_size)+2*validDB_amount
    for trajID in range(validDB_amount):
        steps = len(env_rl_sel_v.ori_traj_set[trajID])
        s = (trajID, 0)
        e = (trajID, steps-1)
        env_rl_sel_v.read(s, e)
        env_rl_sel_v.add_buffer(s, e)
    print('DB size {}, simDB size {}, intial size {} for validation'.format(env_rl_sel_v.DB_size, buffer_valid_size, len(env_rl_sel_v.F_ward)))
    epoch_i_c = 0
    while len(env_rl_sel_v.F_ward) < buffer_valid_size:
        if epoch_i_c % 100 == 0:
            print('time', len(env_rl_sel_v.F_ward), '/', buffer_valid_size)
        observation_inb = env_rl_sel_v.get_agent_inb_state(SampleList_val[epoch_i_c])
        action_inb = RL_INB.online_act(observation_inb)
        env_rl_sel_v.update_inb(action_inb)
        epoch_i_c += 1
    F1 = env_rl_sel_v.submit_query(Rtree_ref_val, test_query)
    return F1
        
def Training():
    batch_size = 32
    check_res = -1
    query_count = 50
    
    for epoch in range(12):
        train_start = time.time()
        env_rl_sel = TrajComp(traj_path, epoch*trainDB_amount, (epoch+1)*trainDB_amount, K, LEN)
        tra_set = env_rl_sel.ori_traj_set
        tra_DB_DISTRI, _, tra_DB_DISTRI_trajID = F.get_distribution_feature_gau(tra_set)
        _, query_part1, _ = F.get_query_workload_gau(tra_DB_DISTRI)
        train_query = query_part1
        
        Rtree_ref_tra, _ = F.build_Rtree(tra_set, traj_path+'tra_tree_'+str(epoch*trainDB_amount)+'_'+str((epoch+1)*trainDB_amount))
        print('bounds', Rtree_ref_tra.idx.get_bounds())
        Rtree_ref_tra.idx.close()
        Rtree_ref_tra = F.obtain_Rtree(traj_path+'tra_tree_'+str(epoch*trainDB_amount)+'_'+str((epoch+1)*trainDB_amount))
        print('re-check bounds', Rtree_ref_tra.idx.get_bounds())
        buffer_train_size = int(ratio*env_rl_sel.DB_size)+2*trainDB_amount
        
        for epoch_i in range(5):
            Transition_oub, SampleList_tra = [], []
            sam_size = int(env_rl_sel.DB_size*ratio)
            env_rl_sam = TrajSam(tra_set, level_start, 'gau')
            print('done env_rl_sam init')
            for sam_i in range(int(sam_size*1.2)):
                obs_oub = []
                act_oub = []
                transition_oub = []
                observation_oub = env_rl_sam.starting(level_start)
                obs_oub.append(observation_oub)
                while True:
                    action_oub = RL_OUB.act(observation_oub)
                    act_oub.append(action_oub)
                    if action_oub == 8 or env_rl_sam.level_cur == level_end:
                        break
                    observation_oub = env_rl_sam.get_agent_oub_state(action_oub)
                    obs_oub.append(observation_oub)
                    if len(obs_oub) >= 2:
                        transition_oub.append([obs_oub[-2], act_oub[-1], obs_oub[-1], False])
                if len(transition_oub) > 0:
                    transition_oub[-1][-1] = True
                Transition_oub.append(transition_oub)
                SampleList_tra.append(env_rl_sam.cubeID)
            env_rl_sam.get_ID2Grids(SampleList_tra)
            
            print('init the whole database for training with sample list size', len(SampleList_tra))           
            env_rl_sel = TrajComp(traj_path, epoch*trainDB_amount, (epoch+1)*trainDB_amount, K, LEN, tra_DB_DISTRI, env_rl_sam.ID2Grids, 'T') #softcopy
            
            for trajID in range(trainDB_amount):
                steps = len(env_rl_sel.ori_traj_set[trajID])
                s = (trajID, 0)
                e = (trajID, steps-1)
                env_rl_sel.read(s, e)
                env_rl_sel.add_buffer(s, e)
            print('DB size {}, simDB size {}, intial size {} for training'.format(env_rl_sel.DB_size, buffer_train_size, len(env_rl_sel.F_ward)))
        
            obs_inb = []
            act_inb = []
            transition_inb = []
            transition_oub_tmp = []            
            epoch_i_c = 0
            
            while len(env_rl_sel.F_ward) < buffer_train_size:
                transition_oub_tmp.append(Transition_oub[epoch_i_c])
                observation_inb = env_rl_sel.get_agent_inb_state(SampleList_tra[epoch_i_c])
                obs_inb.append(observation_inb)
                action_inb = RL_INB.act(observation_inb)
                env_rl_sel.update_inb(action_inb)
                if len(env_rl_sel.F_ward) == buffer_train_size:
                    done = True
                else:
                    done = False   
                if len(obs_inb) >= 2:
                    transition_inb.append([obs_inb[-2], act_inb[-1], obs_inb[-1], done])
                act_inb.append(action_inb)
                
                # training
                if epoch_i_c % query_count == 0:
                    print(epoch, 'training epoch...', epoch_i, 'at', epoch_i_c)
                    reward = env_rl_sel.submit_query(Rtree_ref_tra, train_query)
                    print('training query performance reward {} on epoch_i_c {}'.format(reward, epoch_i_c))
                    for tr in transition_inb:
                        RL_INB.remember(tr[0], tr[1], reward, tr[2], tr[3])
                    for Tr in transition_oub_tmp:
                        for tr in Tr:
                            if len(tr) > 0:
                                RL_OUB.remember(tr[0], tr[1], reward, tr[2], tr[3])
                    transition_inb.clear()
                    transition_oub_tmp.clear()
                
                if epoch_i_c % query_count == 0 or done:
                    if len(RL_INB.memory) > batch_size:
                        RL_INB.replay(batch_size)
                    if len(RL_OUB.memory) > batch_size:
                        RL_OUB.replay(batch_size)
                epoch_i_c += 1
                
            RL_INB.update_target_model()
            RL_OUB.update_target_model()
            
            # validation
            print('validation...')
            res1 = run_dqn()
            RL_INB.save(model_path+'save/'+globaltime+'_gau_RL_INB_F1_' + str(res1)+'.h5')
            RL_OUB.save(model_path+'save/'+globaltime+'_gau_RL_OUB_F1_' + str(res1)+'.h5')
            print('epoch {} epoch_i {} Save model at scanning trajID {} with query performance {}'.format(epoch, epoch_i, trajID, res1))
            if res1 >= check_res:
                check_res = res1
            print('==>current best model with query performance {}'.format(check_res))
            
if __name__ == "__main__":
    
    traj_path = './TrajData/Geolife_out/'
    model_path = './'
    trainDB_amount = 500
    validDB_amount = 1000
    K = 2
    LEN = 2
    ratio = 0.001
    a_size_inb, s_size_inb = K, K*LEN
    level_start, level_end = 10, 12
    
    #reinforcement_learning
    RL_INB = AGENT_Sel(s_size_inb, a_size_inb)
    RL_OUB = AGENT_Sam(2*8, 9)
    
    start = time.time()
    val_start = 6000
    env_rl_sel_v = TrajComp(traj_path, val_start, val_start + validDB_amount, K, LEN)
    sam_size_v = int(env_rl_sel_v.DB_size*ratio)
    val_set = env_rl_sel_v.ori_traj_set
    env_rl_sam_v = TrajSam(val_set, level_start, 'gau')
    
    #only one time (too slow) and dump then,*can comment the lines after*
    Rtree_ref_val, _ = F.build_Rtree(val_set, traj_path+'val_tree_'+str(val_start)+'_'+str(val_start+validDB_amount))
    print('bounds', Rtree_ref_val.idx.get_bounds())
    Rtree_ref_val.idx.close()
    #only one time (too slow) and dump then,*can comment the lines after*
    
    Rtree_ref_val = F.obtain_Rtree(traj_path+'val_tree_'+str(val_start)+'_'+str(val_start+validDB_amount))
    print('re-check bounds', Rtree_ref_val.idx.get_bounds())
    
    val_DB_DISTRI, _, val_DB_DISTRI_trajID = F.get_distribution_feature_gau(val_set)
    _, query_part1, query_part2 = F.get_query_workload_gau(val_DB_DISTRI)
    
    test_query = query_part1 + query_part2
    
    print('Training...')
    sortedlist = Training()
    print("Training elapsed time = %s", float(time.time() - start))
