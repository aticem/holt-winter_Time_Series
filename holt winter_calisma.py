
##################################################
# Holt-Winters
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# pd.set_option('display.float_format', lambda x: '%0.2' % x)
# np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})
warnings.filterwarnings('ignore')


##################################################
# Data Set
##################################################

data = sm.datasets.co2.load_pandas()
y =  data.data
y.index.sort_values()

y = y['co2'].resample('M').mean()
y.head()

# filling nan values with before ones
y = y.fillna(y.bfill())
y.head()


y.plot(figsize=(15, 6))
plt.show()

# train set =  from 1958 to end of 1997
y.index.sort_values(ascending=True)

train = y[:'1997-12-01']
train.head()
len(train)
# 477

# test set = from first month of 1998 to end of the data
test = y['1998-01-01':]
test.head()
len(test)
# 48

#####################################
# Structural Analysis of Time Series
#####################################


def ts_decompose(y, model="additive", stationary=False):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show()

    if stationary:
        print("HO: Series is not stationary.")
        print("H1: Series is stationary.")
        p_value = sm.tsa.stattools.adfuller(y)[1]
        if p_value < 0.05:
            print(F"Result: Series is stationary ({p_value}).")
        else:
            print(F"Result: Series is not stationary ({p_value}).")


for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, True)

# series is not stationary and uses additive model.
# series has trend and seasonality.


#################################
# FORECASTING WITH SINGLE EXPONENTIAL SMOOTHING
#################################

# It can be used in stationary series. Cannot be used if there are trends and seasonality.
# It can be used but does not give the desired results.


#################################
# Stationary Test (Dickey-Fuller Testi)
#################################

# another way to test stationary status
def is_stationary(y):
    print("HO: Series is not stationary.")
    print("H1: Series is stationary .")
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"result: Series is stationary  ({p_value}).")
    else:
        print(F"result: Series is not stationary ({p_value}).")


is_stationary(y)


#################################
# Single Exponential Smoothing Model
#################################

ses_model =SimpleExpSmoothing(train).fit(smoothing_level=0.5)
y_pred =ses_model.forecast(48)
train.plot(title = "Single Exponential Smoothing")
test.plot()
y_pred.plot()
plot.show()
mean_absolute_error(test, y_pred)
#6.89

#################################
# Optimizing Single Exponential Smoothing
#################################


def optimize_ses(train, alphas, step=48):
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))


alphas = np.arange(0.01, 1, 0.10)
optimize_ses(train, alphas)
# alpha: 0.11 mae: 6.0218


#################################
# Final SES Model
#################################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.11)
y_pred = ses_model.forecast(48)
train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)
# 6.021775307808117

#################################
# FORECASTING WITH DOUBLE EXPONENTIAL SMOOTHING
#################################

# DES: Level + Trend

#################################
# DES Model
#################################

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_slope=0.5)
y_pred = des_model.forecast(48)
train["1985":].plot(title="Double Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)



#################################
# Optimizing Double Exponential Smoothing
#################################

def optimize_des(train, alphas, betas, step=48):
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha,
                                                                     smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)
optimize_des(train, alphas, betas)


def optimize_des(train, alphas, betas, step=48):
    print("Optimizing parameters...")
    results = []
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha,
                                                                     smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            results.append([round(alpha, 2), round(beta, 2), round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "mae"]).sort_values("mae")
    print(results)

optimize_des(train, alphas, betas)
# alpha = 0.01  beta = 0.71   mae = 1.74

#################################
# Final DES Model
#################################

final_des_model = ExponentialSmoothing(train, trend='add').fit(smoothing_level=0.01, smoothing_slope = 0.71)
y_pred = final_des_model.forecast(48)
train.plot(title="Double Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)
# 1.7


#################################
# FORECASTING WITH TRIPLE EXPONENTIAL SMOOTHING (HOLT-WINTERS)
#################################


#################################
# TES Model
#################################

tes_model = ExponentialSmoothing(train, trend='add', seasonal_periods= 12).fit(smoothing_level=0.5,
                                                                               smoothing_slope = 0.5,
                                                                               smoothing_seasonal = 0.5)

y_pred = tes_model.forecast(48)
train.plot(title="Triple Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)
#23.2


#################################
# Optimizing Triple Exponential Smoothing
#################################


alphas = betas = gammas = np.arange(0.20, 1, 0.10)
abg = list(itertools.product(alphas, betas, gammas))
abg[0][2]

def optimize_tes(train, abg, step=48):
    print("Optimizing parameters...")
    results = []
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add",
                                         seasonal="add",
                                         seasonal_periods=12).\
            fit(smoothing_level=comb[0],
                smoothing_slope=comb[1],
                smoothing_seasonal=comb[2])

        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

        results.append([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "mae"]).sort_values("mae", ascending=False)
    print(results)

alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

optimize_tes(train, abg)
# alpha=0.9, beta=0.3, gama=0.1, mae=1.21

#################################
# Final TES Model
#################################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=0.9, smoothing_slope=0.3, smoothing_seasonal=0.1)
y_pred = final_tes_model.forecast(48)
train["1985":].plot(title="Triple Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
mean_absolute_error(test, y_pred)

























