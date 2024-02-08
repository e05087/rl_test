
class Agent:
    def __init__(self, init_balance):
        self.init_balance = init_balance
        self.set_balance(init_balance)
        self.asset = Asset()
        self.tr_cost = 0.001 # assum transaction cost as 10bp per one buy/sell
    
    def set_balance(self, balance):
        self.balance = balance

    def set_asset(self, cls, qty):
        self.asset[cls] = qty
    
    def get_asset(self, cls):
        return self.asset[cls]
    
    def rebalance(self, target_mp, price_df):
        current_expected_balance = self.balance
        for cls in target_mp:
            current_expected_balance += self.get_asset(cls) * price_df[cls] * (1 - self.tr_cost)
        
        for cls in target_mp:
            target_balance = current_expected_balance * target_mp[cls]
            target_qty = int(target_balance // (price_df[cls]*(1+self.tr_cost)))
            need_tr_qty = target_qty - self.get_asset(cls)
            if need_tr_qty >= 0:
                tr_cost = 1 + self.tr_cost
            else:
                tr_cost = 1 - self.tr_cost
            self.set_balance(self.balance - price_df[cls] * need_tr_qty * tr_cost)
           
            self.set_asset(cls, target_qty)
        assert self.balance >= 0
        
        current_expected_balance = self.balance
        
        for cls in self.asset:
            current_expected_balance += self.get_asset(cls) * price_df[cls] * (1 - self.tr_cost)
        return current_expected_balance

    def cal_mdd(self, df):
        max_price = df.rolling(12, min_periods=1).max()
        dd = (df/max_price - 1)
        mdd = dd.rolling(12, min_periods=1).min().min()
        return mdd
    
    def cal_cagr_sharpe_mdd_plusratio(self, df):
        result_df_mon = df['expected_balance']
        result_df_year = df.iloc[::12,:]['expected_balance']
        num_plus = (result_df_mon>self.init_balance).sum()
        plusratio = num_plus / len(result_df_mon.index)
        
        mdd = self.cal_mdd(result_df_mon)
        cagr = (result_df_year.iloc[-1]/self.init_balance)**(1/(len(result_df_year.index)-1)) - 1

        ret_df = result_df_year.pct_change()
        ret_df.dropna(inplace=True)
        sharpe = ret_df.mean() / ret_df.std()
        return cagr, sharpe, mdd, plusratio


class Asset(dict):
    def __init__(self):
        self['A1'] = 0
        self['A2'] = 0
        self['A3'] = 0

    def to_list(self):
        return [self['A1'], self['A2'], self['A3']]
    
    def from_array(self, array):
        for idx, i in enumerate(array):
            self['A{}'.format(idx+1)] = i
        return self
