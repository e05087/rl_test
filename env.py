import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from utils import Agent, Asset

class Env(gym.Env):
    def __init__(self, momentum_dict, data_df):
        super(Env, self).__init__()
        self.agent = Agent(10000)
        self.momentum_dict = momentum_dict
        self.data_df = data_df
        self.current_step = 0
        self.base_asset_index = 1
        self.base_asset_ratio = 0.5
        self.org_data_df = data_df
        self.current_dt = self.data_df.iloc[self.current_step]['dt']

        # actions
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.float32)

    def _next_observation(self):

        momentum_1 = self.momentum_dict['mon1'].loc[self.current_dt].to_numpy()
        momentum_3 = self.momentum_dict['mon3'].loc[self.current_dt].to_numpy()
        momentum_6 = self.momentum_dict['mon6'].loc[self.current_dt].to_numpy()

        obs = np.array([momentum_1, momentum_3, momentum_6])

        return obs
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def action_to_mp(self, action):
        action = self.softmax(action)
        action = action * (1 - self.base_asset_ratio)
        action[self.base_asset_index] += self.base_asset_ratio
        return Asset().from_array(action)

    def step(self, action):
        mp = self.action_to_mp(action)
        
        self.current_dt = self.data_df.iloc[self.current_step]['dt']
        expected_balance = self.agent.rebalance(mp, self.data_df.iloc[self.current_step])
        
        self.data_df.at[self.current_step, 'expected_balance'] = expected_balance
        
        self.current_step += 1

        done = False
        reward = 0
        if self.current_step > len(self.data_df.index) -1:
            self.current_step = 0
            done = True
            cagr_w = 0
            sharpe_w = 1
            mdd_w = 0
            plusratio_w = 0
            cagr, sharpe, mdd, plusratio = self.agent.cal_cagr_sharpe_mdd_plusratio(self.data_df)
            reward = cagr * cagr_w + sharpe * sharpe_w + mdd * mdd_w + plusratio * plusratio_w
            self.cagr = cagr
            self.sharpe = sharpe
            self.mdd = mdd
            self.plusratio = plusratio
        
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.agent = Agent(10000)
        self.data_df = self.org_data_df
        self.cagr = 0
        self.sharpe = 0
        self.mdd = 0
        self.plusratio = 0
        self.current_step = 0
        self.current_dt = self.data_df.iloc[self.current_step]['dt']
        return self._next_observation()

    def render(self, mode='human', close=False):
        print("-----------------")
        print("Step\tBalance\tSharpe\tCAGR")
        print("{}\t{}\t{}\t{}".format(self.current_step, self.agent.balance, self.sharpe, self.cagr))