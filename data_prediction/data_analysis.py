import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from io import BytesIO, StringIO

def normality_test(data):
    
    stat, p = stats.normaltest(data)
    print('Statistics = {}, p={}'.format(stat, p))
    alpha = 0.05
    if p > alpha:
        print('Data looks Gaussian (fail to reject H0)')
    else:
        print('Data do not look Gaussian (reject H0)')
    # normaltest returns a 2-tuple of the chi-squred statistic, and the associated p-value
    # if the p-val is very small, it means it is unlikely that the data came from a normal distribution
    
    return stat, p

def kurtosis_skew_test(data):
    print('Kurtois of normal distribution :{}'. format(stats.kurtosis(data)))
    print('Skewness of normal distribution :{}'. format(stats.skew(data)))
    # Kurtois is a measure of the 'tailedness' of the probability distribution of a real-valued random variable
    # A value close to 0 for Kurtosis indicates a normal distribution where asysmmetrical nature is signified by a value between -0.5 and +0.5
    # Moderate skewness refers to the value between -1 and -0.5 or 0.5 and 1
    

def adfuller_test(series, signif=0.05, name='', verbose=False):
    from statsmodels.tsa.stattools import adfuller
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'p_value':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':round(r[3])}
    p_value = output['p_value']
    def adjust(val, length=6): return str(val).ljust(length)
    
    print(' Augmented Dickey-Fuller Test on {}',format(name))
    print(' Null HYpothesis: Data has unit root. Non-Stationary.' )
    print(' Significance level = {}'.format(signif))
    print(' Test Statistic     = {}'.format(output["test_statistic"]))
    print(' No. Lags Chosen    = {}'.format(output["n_lags"]))
    
    for key, val in r[4].items():
        print(' Critical value {} = {}'.format(adjust(key), round(val, 3) ))
    if p_value <=signif:
        print('=> P-Value = {}. Rejecting Null Hypothesis.'.format(p_value))
        print('=> Series is Stationary.')
    else:
        print('=> P-Value = {}. Weak evidence to reject the Null Hypothesis.'.format(p_value))
        print('=> Series is Non-Stationary.')

import seaborn as sns
def plot_data_distribution(data, feature):
    
    import seaborn as sns
    fig = plt.figure()
    ax1 =fig.add_subplot(1, 2, 1)
    sns.distplot(data[feature],hist=True,  bins =10,  hist_kws={'edgecolor':'gray'}, color='blue',  ax= ax1)
    ax1.set_title(feature+' Data Distribution')
    
    ax2 =fig.add_subplot(1, 2, 2)
    stats.probplot(data[feature], plot=ax2)
    ax2.set_title(feature+' Best-fit line for data')
    
    fig.tight_layout()
    
    """
    stats.probplot(data[feature], plot=plt)
    """
    buf  = BytesIO()
    fig.savefig(buf, format='png', dpi=112)
    buf.seek(0)
    
    return buf

def plot_acf_pacf(data, feature):
    test_feature = data[feature]
    #print(data)
    
    fix, ax = plt.subplots(2, figsize=(10, 6))
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    ax[0] = plot_acf(test_feature, ax=ax[0])
    ax[1] = plot_pacf(test_feature, ax=ax[1])
    
    buf  = BytesIO()
    plt.savefig(buf, format='png', dpi=112)
    buf.seek(0)
    
    return buf
