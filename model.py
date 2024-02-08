import pandas as pd
import os
import numpy as np
from utils import Agent, Asset
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from env import Env


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Model:
    def __init__(self, df_dict, init_balance = 10000):
        self.df_dict = df_dict
        self.init_balance = init_balance
        
    def get_random_mp(self):
        asset = Asset()
        mp = np.random.rand(len(asset))
        mp /= mp.sum()
        asset = asset.from_array(mp)
        return asset

    def train_or_load_model(self, momentum_dict):
        model_dir = '{}/model/model'.format(ROOT_DIR)
        env = DummyVecEnv([lambda: Env(momentum_dict, self.df_dict['train'])])
        if not os.path.isfile('{}.zip'.format(model_dir)): # should train
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, tensorboard_log="./rl_tb_log/")
            max_mean_reward = 0
            for i in range(5000):
                model.learn(total_timesteps=50000)
                mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic = True)
                with open('reward.csv','a') as f:
                    f.write("{:.4f}\n".format(mean_reward))
                if max_mean_reward < mean_reward:
                    max_mean_reward = mean_reward
                    model.save(model_dir)
        else:
            model = PPO.load(model_dir)

        self.rl_env = Env(momentum_dict, self.df_dict['train'])
        self.rl_model = model
        return model


    def cal_cagr_sharpe_mdd_plusratio(self, df):
        agent = Agent(self.init_balance)
        return agent.cal_cagr_sharpe_mdd_plusratio(df)

    def backtest(self, data, mode='train', strategy='base'):
        data_df = self.df_dict[mode]

        agent = Agent(self.init_balance)
        
        for reb_idx in data_df.index:
            reb_df = data_df.iloc[reb_idx]
            if strategy == 'base':
                expected_balance = agent.rebalance(data, reb_df)
            elif strategy == 'adm':
                mp = Asset()
                signal = data.loc[reb_df['dt']]
                print(signal)
                mp[signal] = 1.0
                expected_balance = agent.rebalance(mp, reb_df)
            elif strategy == 'rl':
                assert self.rl_model is not None
                momentum_1 = data['mon1'].loc[reb_df['dt']].to_numpy()
                momentum_3 = data['mon3'].loc[reb_df['dt']].to_numpy()
                momentum_6 = data['mon6'].loc[reb_df['dt']].to_numpy()
                obs = np.array([momentum_1, momentum_3, momentum_6])
                action, _ = self.rl_model.predict(obs, deterministic=True)
                mp = self.rl_env.action_to_mp(action)
                print(reb_df['dt'], mp)
                expected_balance = agent.rebalance(mp, reb_df)

            else:
                assert False
            
            data_df.at[reb_idx, 'expected_balance'] = expected_balance
                

        return data_df
    
    
if __name__ == '__main__':
    
    csv_path = "{}/analyst_data.csv".format(ROOT_DIR)
    df = pd.read_csv(csv_path)
    df['year'] = pd.to_datetime(df['dt']).dt.year
    df['month'] = pd.to_datetime(df['dt']).dt.month
    df['day'] = pd.to_datetime(df['dt']).dt.day
    df = df.sort_values(by=['year','month','day']).drop_duplicates(subset=['year', 'month'], keep='last')
    
    df = df.reset_index()
    df = df.drop(["year", "month", "day", "index"], axis=1)

    df_momentum = df.set_index("dt")
    momentum_1 = df_momentum.pct_change(periods=1)
    momentum_1.dropna(inplace= True)
    momentum_3 = df_momentum.pct_change(periods=3)
    momentum_3.dropna(inplace= True)
    momentum_6 = df_momentum.pct_change(periods=6)
    momentum_6.dropna(inplace= True)

    momentum_dict = dict()

    momentum_dict['mon1'] = momentum_1
    momentum_dict['mon3'] = momentum_3
    momentum_dict['mon6'] = momentum_6

    adm = momentum_1 + momentum_3 + momentum_6
    adm.dropna(inplace= True)
    adm = adm.idxmax(axis=1)



    train_start = '2004-03-31'
    test_start = '2017-01-01'

    assert train_start < test_start
    df_dict = dict()
    df_dict['train'] = df[(df['dt'] >= train_start) & (df['dt'] < test_start)].reset_index()
    df_dict['test'] = df[(df['dt'] >= test_start)].reset_index()



    model = Model(df_dict)

    target_mp = Asset()
    max_sharpe = 0
    max_mp = target_mp.to_list()
    max_cagr = 0
    max_result = 0

    model.train_or_load_model(momentum_dict)
    result = model.backtest(momentum_dict, mode='train', strategy='rl')
    cagr, sharpe, mdd, plusratio = model.cal_cagr_sharpe_mdd_plusratio(result)
    print("[RL]", cagr,sharpe,mdd,plusratio)
    
    
    ## using efficient frontier, find best sharpe ratio portfolio in train period
    for i in range(10000):
        target_mp = model.get_random_mp()
        result = model.backtest(target_mp, mode='train', strategy='base')
        cagr, sharpe, mdd, plusratio = model.cal_cagr_sharpe_mdd_plusratio(result)
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            max_mp = target_mp.to_list()
            max_cagr = cagr
            max_result = result
    #print("[EF]", max_cagr,max_sharpe,max_mp)

    result = model.backtest(Asset().from_array(max_mp), mode='test', strategy='base')
    cagr, sharpe, mdd, plusratio = model.cal_cagr_sharpe_mdd_plusratio(result)
    print("[EF]", cagr,sharpe,mdd,plusratio, max_mp)
    result = model.backtest(adm, mode='test', strategy='adm')
    cagr, sharpe, mdd, plusratio = model.cal_cagr_sharpe_mdd_plusratio(result)
    print("[ADM]", cagr,sharpe,mdd,plusratio)
