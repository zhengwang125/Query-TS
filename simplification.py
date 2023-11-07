# -*- utf-8 -*-
from rl_env import TrajComp, TrajSam
import time
from rl_brain import AGENT_Sel, AGENT_Sam
import warnings
import data_utils as F
import random
import pickle
random.seed(0)
warnings.filterwarnings("ignore")

def run_dqn():
    SampleList_val = []
    for sam_i in range(sam_size_v):
        observation_oub = env_rl_sam_v.starting(level_start)
        while True:
            action_oub = RL_OUB.fast_online_act(observation_oub)
            if action_oub == 8 or env_rl_sam_v.level_cur == level_end:
                break
            observation_oub = env_rl_sam_v.get_agent_oub_state(action_oub)
        SampleList_val.append(env_rl_sam_v.cubeID)
    env_rl_sam_v.get_ID2Grids(SampleList_val)
    
    env_rl_sel_v = TrajComp(traj_path, val_start, val_start + validDB_amount, K, LEN, '', env_rl_sam_v.ID2Grids, 'V')
    buffer_valid_size = int(ratio*env_rl_sel_v.DB_size)+2*validDB_amount
    for trajID in range(validDB_amount):
        steps = len(env_rl_sel_v.ori_traj_set[trajID])
        s = (trajID, 0)
        e = (trajID, steps-1)
        env_rl_sel_v.read(s, e)
        env_rl_sel_v.add_buffer(s, e)
    epoch_i_c = 0
    while len(env_rl_sel_v.F_ward) < buffer_valid_size:
        observation_inb = env_rl_sel_v.get_agent_inb_state(SampleList_val[epoch_i_c])
        action_inb = RL_INB.fast_online_act(observation_inb)
        env_rl_sel_v.update_inb(action_inb)
        epoch_i_c += 1

    sim_DB = env_rl_sel_v.get_simplified_DB()
    pickle.dump(sim_DB, open(traj_path+'/'+label+'_queryts_'+str(val_start)+'_'+str(val_start+validDB_amount)+'_'+str(ratio)+'_geolife', 'wb'), protocol=2)
            
if __name__ == "__main__":

    label = 'data'
    traj_path = './TrajData/Geolife_out/'
    model_path = './'
    validDB_amount = 1000
    val_start = 7000 
    K = 2
    LEN = 2
    a_size_inb, s_size_inb = K, K*LEN
    level_start, level_end = 9, 12

    RL_INB = AGENT_Sel(s_size_inb, a_size_inb)
    RL_OUB = AGENT_Sam(2*8, 9)
    
    val_start = 7000
    env_rl_sel_v = TrajComp(traj_path, val_start, val_start + validDB_amount, K, LEN)
   
    val_set = env_rl_sel_v.ori_traj_set
    env_rl_sam_v = TrajSam(val_set, level_start, label)
    
    RL_INB.load(model_path+'save/model_'+str(label)+'_RL_INB_F1.h5')
    RL_OUB.load(model_path+'save/model_'+str(label)+'_RL_OUB_F1.h5')
    
    running_time = []
    for ratio in [0.0025]: 
        sam_size_v = int(env_rl_sel_v.DB_size*ratio)
        run_dqn()
    
    #############################################################################################################
    
    label = 'gau'
    traj_path = './TrajData/Geolife_out/'
    model_path = './'
    validDB_amount = 1000
    val_start = 7000 
    K = 2
    LEN = 2
    a_size_inb, s_size_inb = K, K*LEN
    level_start, level_end = 10, 12

    RL_INB = AGENT_Sel(s_size_inb, a_size_inb)
    RL_OUB = AGENT_Sam(2*8, 9)
    
    val_start = 7000
    env_rl_sel_v = TrajComp(traj_path, val_start, val_start + validDB_amount, K, LEN)
   
    val_set = env_rl_sel_v.ori_traj_set
    env_rl_sam_v = TrajSam(val_set, level_start, label)
    
    RL_INB.load(model_path+'save/model_'+str(label)+'_RL_INB_F1.h5')
    RL_OUB.load(model_path+'save/model_'+str(label)+'_RL_OUB_F1.h5')
    
    running_time = []
    for ratio in [0.0025]: 
        sam_size_v = int(env_rl_sel_v.DB_size*ratio)
        run_dqn()
