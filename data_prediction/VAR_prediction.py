import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class prediction():
    def __init__(self, maxlag):
        self.maxlag= maxlag
    
    def model_VAR(self, data):
        from statsmodels.tsa.api import VAR
        model= VAR(data)
        #model_result = model.fit(maxlags=15, ic='aic')
        model_result = model.fit(maxlags=self.maxlag)
        print("VAR Result:\n", model_result.summary())
        return model_result
    
    def get_durbin_watson_result(self, model_result, train):
        from statsmodels.stats.stattools import durbin_watson
        out_table = durbin_watson(model_result.resid)
        for col, val in zip(train.columns, out_table):
            print((col), ':', round(val,2))
        return out_table
    
    def cointegration_test(self, train, alpha=0.05):
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        out = coint_johansen(train, -1, 5)
        d ={'0.90':0, '0.95':1, '0.99':2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1-alpha)]]

        def adjust(val, length=len(train)): return str(val).ljust(length)

        print('Name :: Test Stat > C (95%) => Signif    \n', '--'*20)
        for col, trace, cvt in zip(train.columns, traces, cvts):
            print(adjust(col), '::' , adjust(round(trace, 2),9) ,">", adjust(cvt, 8), '=>', trace>cvt)

    def model_prediction(self, model_result, train, test, n_obs):
        lag_order = model_result.k_ar
        # input
        input_data = train.values[-lag_order:]
        #forecassting
        pred_result = model_result.forecast(input_data, n_obs)
        pred = pd.DataFrame(pred_result, index = test.index, columns = test.columns)
        return pred
    

    
    def get_metrics_table(self, origin, prediction):
        import sklearn.metrics as metrics
        metrics_table = pd.DataFrame(index=origin.columns, columns =['accuracy', 'mae', 'mse', 'msle', 'rmse', 'rmsle', 'r2'])
        for feature in metrics_table.index:
            X1 = origin[feature]
            X2 = prediction[feature]
            accuracy = (100-abs(X1 -X2)/X1*100).mean().round(2)
            mae = metrics.mean_absolute_error(X1, X2).round(2)
            mse = metrics.mean_squared_error(X1, X2).round(2)
            msle = metrics.mean_squared_log_error(X1, X2).round(2)
            rmse = np.sqrt(mse).round(2)
            rmsle = np.log(rmse).round(2)
            r2=metrics.r2_score(X1, X2).round(2)
            metrics_table.loc[feature] = [accuracy, mae, mse, msle, rmse,rmsle, r2]
        return metrics_table
    
