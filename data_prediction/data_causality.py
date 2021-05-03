from io import BytesIO, StringIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
def timelag_correlation(data, d1_feature, d2_feature, lag_range):
     #time lagged cross correlation
    d1 = data[d1_feature]
    d2 = data[d2_feature]  
    rs = [d1.corr(d2.shift(lag)) for lag in range(-lag_range, lag_range+1)]
    
    title ='Influence of '+ d1_feature +' on '+d2_feature +'\n'
    title = title + 'Maximum abs correlation value: '+ str(rs[np.argmax(np.abs(rs))].round(2))  +'\n'   
    title = title + 'Maximum - Minimum correlation value: '+ str((max(rs)-min(rs)).round(2))+'\n'          
                          
    return rs, title


def timelag_correlation_fig(rs, lag_range, d1_feature, d2_feature, title):
    
    f, ax = plt.subplots(figsize=(16, 7))
    ax.plot(rs)
    ax.axvline(int(len(rs)/2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(np.abs(rs)), color='r', linestyle='--', label='Peak Synchrony')
    
    ax.set(title=title, ylim=[-1, 1], xlabel = 'offset', ylabel ='Pearson r')
    x_ticks = [x*2 for x in range (0, int(lag_range)+1)]
    x_ticks_label = [x*2-lag_range for x in range (0, int(lag_range)+1)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_label)
    plt.legend()
    buf  = BytesIO()
    plt.savefig(buf, format='png', dpi=112)
    buf.seek(0)
    
    return buf

def timelag_correlation_fig_ipython(rs, lag_range, title):
    
    f, ax = plt.subplots(figsize=(16, 7))
    ax.plot(rs)
    ax.axvline(int(len(rs)/2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(np.abs(rs)), color='r', linestyle='--', label='Peak Synchrony')
    
    ax.set(title=title, ylim=[-1, 1], xlabel = 'offset', ylabel ='Pearson r')
    x_ticks = [x*2 for x in range (0, int(lag_range)+1)]
    x_ticks_label = [x*2-lag_range for x in range (0, int(lag_range)+1)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_label)
    
    plt.legend()
    plt.show()

def get_specific_rs_information(rs):
    max_idx  = np.argmax(rs)
    min_idx =np.argmin(rs)
    absmax_idx = np.argmax(np.abs(rs))
    absmax = rs[absmax_idx].round(2)
    min_max_diff = (max(rs)-min(rs)).round(2)
    best_lag = int(len(rs)/2) -  np.argmax(np.abs(rs))

    return max_idx, min_idx, absmax_idx, absmax, min_max_diff, best_lag


def grangers_causation_matrix(maxlag, data ,variables, test='ssr_chi2test', verbose=False):
    table = pd.DataFrame(np.zeros((len(variables), len(variables))), columns =variables, index=variables)
    for c in table.columns:
        for r in table.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: 
                print('Y =',r, 'X=',c, 'P Values=',p_values)
            min_p_value = np.min(p_values)
            table.loc[r, c] = min_p_value
    table.columns=[var+'_x' for var in variables]
    table.index = [var+'_y' for var in variables]
    ####
    print("Grangers Test Result: \n", table,'\n')
    return table

def plot_causal_table(table, feature):
    
    fig = plt.figure()
    ax =fig.add_subplot(1, 1, 1)
    
    test_plot_data = table.loc[feature].sort_values()
    test_plot_data.plot(grid=True, figsize=(20, 6))
    plt.axhline(y=0.05, color='r', linewidth=1, linestyle='--')
    
    ax.set_title(feature+' Granger Causality Test Result')
    fig.tight_layout()
    
    

    buf  = BytesIO()
    plt.savefig(buf, format='png', dpi=112)
    buf.seek(0)
    
    return buf

def get_causal_history_table(data, d1_feature_list, d2_feature_list, lag_range):
    best_lag_history=pd.DataFrame()
    for d1_feature in d1_feature_list:
        for d2_feature in d2_feature_list:
            if d1_feature != d2_feature:
                rs, title = timelag_correlation(data, d1_feature, d2_feature, lag_range)
                #dc.timelag_correlation_fig_ipython(rs, lag_range, title)
                max_idx, min_idx, absmax_idx, absmax, min_max_diff, best_lag = get_specific_rs_information(rs)
                best_lag_df = pd.DataFrame({'d1':[d1_feature], 'd2':[d2_feature], 'lag':[best_lag], 
                                            'max_corr':[absmax], 'max_diff_corr':[min_max_diff]})
                best_lag_history = best_lag_history.append(best_lag_df)
    causal_history_table = best_lag_history.iloc[(-np.abs(best_lag_history['max_corr'].values)).argsort()]
    return causal_history_table


def get_high_causal_table(history_table, factor_list):
    result =pd.DataFrame()
    
    p1 = 0.4
    p2 = 0.6
    p3 = 0.5
    
   # p1 = p2 = p3 = p4 =0.3
    
    for factor_2 in factor_list:
        temp = history_table[history_table.d2 == factor_2]
        temp = temp.iloc[(-np.abs(temp['max_diff_corr'].values)).argsort()]

        temp1 = temp[(temp['max_corr'].abs()> p1)&(temp['max_diff_corr'].abs()> p2)]
        temp2 = temp[(temp['max_corr'].abs()> p3)&(temp['max_corr'].abs() <= p1)]
        
        data_list = [result, temp1, temp2]
        result = pd.concat(data_list, sort=True)
    print(result)
    return result


def get_high_causal_dict(result):
    d2_list = list(set(result['d2']))
    causal_dict={}
    for factor_2 in d2_list :
        temp = result[result['d2']==factor_2]
        d1_list = list(temp['d1'].values)
        causal_dict[factor_2]=d1_list
        causal_dict[factor_2+'_corr'] = list(temp['max_corr'].astype(str).values)
        causal_dict[factor_2+'_lag'] = list(temp['lag'].astype(str).values)
        
    return causal_dict