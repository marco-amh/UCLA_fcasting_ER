# -*- coding: utf-8 -*-
# Capstone Project
# Marco Antonio Martinez Huerta
# Faculty Advisor: Aaron Tornell

# 0.1 Import modules

import os
os.chdir('C://Users//marco//Desktop//UCLA//Capstone_Project')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import warnings
import statsmodels.formula.api as smf
import scipy
warnings.filterwarnings("ignore")

# 1. Functions

## 1.1. API: Banco de México


def Banxico(serie,name, plot):
    API_key_banxico = "bafc26d5c3c13e17b465d9550a5f38ca5f888a4dc8c6661ef3658720ba1702a9"
    URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
    parameters = str(serie) + '/datos/?token=' + str(API_key_banxico)
    PATH =  str(URL) + parameters
    r = requests.get(PATH)
    data =r.text
    data = json.loads(data)

    list_a = data["bmx"]["series"][0]['datos']
    for i in list_a:
        if i['dato'] == "N/E":
            pass
        else:
            nuevo_val = float(i['dato'].replace(",",""))
            i['dato'] = nuevo_val

    data = []
    date = []

    for i in range(0,len(list_a)):
        if list_a[i]['dato'] == "N/E":
            pass
        else:
            date.append(list_a[i]['fecha'])
            data.append(list_a[i]['dato'])

    df = pd.DataFrame(date)
    df['Date'] = pd.DataFrame(date)
    df[name] = pd.DataFrame(data)
    df = df.set_index('Date')
    df = df.drop(columns=[0])
    df.index = pd.to_datetime(df.index, format = '%d/%m/%Y').strftime('%Y-%m-%d')
    df[name] = df[name].astype('float')
    if plot == True:
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
        plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
        plt.xticks(np.arange(0, len(df), step=24), rotation=90)
        plt.show() #plot
    
    return(df)

## 1.2. API: FRED

def Fred(serie,name, plot):
    API_key_fred = "1d956f22f710e02c387275323899460a"
    URL = "https://api.stlouisfed.org/fred/series/observations?series_id="
    parameters = str(serie) + '&api_key=' + str(API_key_fred) + "&file_type=json"
    PATH =  str(URL) + parameters
    r = requests.get(PATH)
    data = r.text
    data = json.loads(data)
    list_a = data['observations']
    data = []
    date = []
    for i in range(0,len(list_a)):
        date.append(list_a[i]['date'])
        data.append(list_a[i]['value'])
    
    df = pd.DataFrame(date)
    df['Date'] = pd.DataFrame(date)
    df[name] = pd.DataFrame(data)
    df = df.set_index('Date')
    df = df.drop(columns=[0])
    df.index = pd.to_datetime(df.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')
    df.sort_index(ascending=True, inplace=True)
    df[name] = df[name].astype('float')
    if plot == True:
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
        plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
        plt.xticks(np.arange(0, len(df), step=24), rotation=90)
        plt.show() #plot
    return(df)

## 1.3.a. Kalman Filter

def Kalman_Filter(x,j,p,v, roll, sd,plot):
    df = pd.DataFrame(x)
    x = x[x.columns[0]]
    # Initial parameters
    fcast = [] 
    fcast_error = []
    kgain = []
    p_tt = []
    x_tt=[]
    filter_error = []
    n = len(x)
    mu = 0 #mean
    a = j #less than 1
    x0 = x11 = 0 #initial x_0=0. This is for part A in this question
    p11 = p00 = 1 #initial variance state model
    v = v # variance state space model
    w = p # variance state representation

    for i in range(0,n):
        x10 = mu + (x11 * a) # 1st step: Estimation "state x_t" equation. Its equal to y_hat
        p10 = (a**2 * p11 ) + w # 2nd step: variance of the state variable
        fcast.append((a**2) * x10)
        fcast_error.append( x[i] - ((a**2) * x10) )# forecast error
        feta = p10 + v # add variance residual v
        k = p10 * (1/feta)
        kgain.append(p10 * (1/feta)) # Kalman Gain
        x11 = ((1-k) * (x10*a)) + (k * x[i]) # filtered state variable
        p11 = p10 - k * p10 #  variance of state variable
        p_tt.append(p11) #  variance of state variable
        x_tt.append(x11) # store filters state variable
        filter_error.append(x[i] - x11) # residual
    
    gap = x-x_tt
    x = x
    df[str(df.columns[0])+'_filtered'] = x_tt
    df[str(df.columns[0])+'_gap'] = gap
    df[str(df.columns[0])+'_variance_state'] = p_tt
    df[str(df.columns[0])+'_gain'] = kgain
    df[str(df.columns[0])+'_fcast_h1'] = fcast
    df[str(df.columns[0])+'_std_roll+']=0
    for i in range(roll,len(x)):
        df[str(df.columns[0])+'_std_roll+'][i:i+1] = (x[i-roll:i].std())*sd
    df[str(df.columns[0])+'_std_roll-']=0
    for i in range(roll,len(x)):
        df[str(df.columns[0])+'_std_roll-'][i:i+1] = ((x[i-roll:i].std())*sd)*-1
    
    
    print('the kalman gain is: ', str(kgain[-1]))
    if plot == True:
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(x.index,x_tt, lw=2, linestyle='-', color='b')
        line, = plt.plot(x.index,x, lw=2, linestyle='-', color='r')
        plt.gca().set(title='kalman', xlabel = 'Date', ylabel = 'variable')
        plt.xticks(np.arange(0, len(x), step=24), rotation=90)
        #plt.show() #plot
        
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(x.index,gap, lw=2, linestyle='-', color='y')
        plt.axhline(y = 0.0, color='r', linestyle='-')
        plt.gca().set(title='Kalman Gap', xlabel = 'Date', ylabel = 'Gap')
        plt.xticks(np.arange(0, len(x), step=24), rotation=90)
        plt.show() #plot
        
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(x.index,gap, lw=2, linestyle='-', color='y')
        line, = plt.plot(x.index,df[str(df.columns[0])+'_std_roll+'], lw=2, linestyle='-', color='b')
        line, = plt.plot(x.index,df[str(df.columns[0])+'_std_roll-'], lw=2, linestyle='-', color='b')
        plt.axhline(y = 0.0, color='r', linestyle='-')
        plt.gca().set(title='Kalman Gap and Rolling Standard Deviation', xlabel = 'Date', ylabel = 'Gap')
        plt.xticks(np.arange(0, len(x), step=24), rotation=90)
        plt.show() #plot
    return df


## 1.3.b. Kalman Filter simple version

def KF(x,k, roll, sd,plot):
    df = pd.DataFrame(x)
    x = x[x.columns[0]]
    # Initial parameters
    x_tt=[]
    filter_error = []
    n = len(x)
    x11 = 0 #initial x_0=0. This is for part A in this question


    for i in range(0,n):
        x11 = ((1-k) * (x11)) + (k * x[i]) # filtered state variable
        x_tt.append(x11) # store filters state variable
        filter_error.append(x[i] - x11) # residual
    
    gap = x-x_tt
    x = x
    df[str(df.columns[0])+'_filtered'] = x_tt
    df[str(df.columns[0])+'_gap'] = gap
    df[str(df.columns[0])+'_std_roll+']=0
    for i in range(roll,len(x)):
        df[str(df.columns[0])+'_std_roll+'][i:i+1] = (x[i-roll:i].std())*sd
    df[str(df.columns[0])+'_std_roll-']=0
    for i in range(roll,len(x)):
        df[str(df.columns[0])+'_std_roll-'][i:i+1] = ((x[i-roll:i].std())*sd)*-1
    
    
    if plot == True:
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(x.index,x_tt, lw=2, linestyle='-', color='b')
        line, = plt.plot(x.index,x, lw=2, linestyle='-', color='r')
        plt.gca().set(title='kalman', xlabel = 'Date', ylabel = 'variable')
        plt.xticks(np.arange(0, len(x), step=24), rotation=90)
        #plt.show() #plot
        
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(x.index,gap, lw=2, linestyle='-', color='y')
        plt.axhline(y = 0.0, color='r', linestyle='-')
        plt.gca().set(title='Kalman Gap', xlabel = 'Date', ylabel = 'Gap')
        plt.xticks(np.arange(0, len(x), step=24), rotation=90)
        plt.show() #plot
        
        plt.figure(figsize=(16,8))
        plt.style.use('ggplot')
        line, = plt.plot(x.index,gap, lw=2, linestyle='-', color='y')
        line, = plt.plot(x.index,df[str(df.columns[0])+'_std_roll+'], lw=2, linestyle='-', color='b')
        line, = plt.plot(x.index,df[str(df.columns[0])+'_std_roll-'], lw=2, linestyle='-', color='b')
        plt.axhline(y = 0.0, color='r', linestyle='-')
        plt.gca().set(title='Kalman Gap and Rolling Standard Deviation', xlabel = 'Date', ylabel = 'Gap')
        plt.xticks(np.arange(0, len(x), step=24), rotation=90)
        plt.show() #plot
    return df



## 1.4. Sharpe ratio

def sharpe_ratio(returns, rfr):
    excess_returns = returns - rfr
    n = len(excess_returns)
    mean = excess_returns.mean()
    std = np.sqrt(np.sum(np.square(excess_returns-mean))/(n-1))
    return(mean/std)

## 1.5. Skewness

def skewness(returns):
    n = len(returns)
    mean = returns.mean()
    std = np.sqrt(np.sum(np.square(returns))/(n-1))
    skewness = np.mean(np.power((returns - mean)/std, 3))
    return(skewness)

## 1.6. Gini index

def Gini_index(x):
    x = x.to_frame().dropna().reset_index()
    x=x.drop(columns=x.columns[0])
    x.rename(columns={x.columns[0]: 'col1' }, inplace=True)
    x = x.sort_values(by='col1', ascending=True)
    n = int(len(x))
    cum_x = np.cumsum(x)
    cum_x1 = cum_x.copy()
    cum_x2 = cum_x.copy()

    cum_x1[cum_x1 < 0] = 0
    cum_x2[cum_x2 > 0] = 0

    slope = np.cumsum(x).max()/int(n)

    area_1 = int(n)*(np.cumsum(x).max())/2

    mtx=[]
    j=1
    for i in range(0,len(x)):
        mtx.append(slope*j-cum_x1[i:i+1])
        j=j+1

    val = pd.concat(mtx)
    lorenz_1 = np.sum(val)
    lorenz_2 = abs(np.sum(cum_x2))
    area_2 = 0-cum_x.min()*len(cum_x)

    gini=(lorenz_1+lorenz_2)/(area_1+area_2)
    return gini

## 1.7. Mean Squared Error function

def mse(df):
    x = df.dropna().to_numpy()
    n = x.shape[0]
    mse = (np.sum(x**2))/n
    return(mse)

## 1.8.Plot histograms

def plot_returns(df, column, title):
    mu = df[column].mean()
    sigma = df[column].std()
    n, bins, patches = plt.hist(x=df[column], bins='auto', 
                                color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel(None)
    plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if (maxfreq % 10 > 0) else maxfreq + 10)

## 1.9. Geometric Average Return
def geo_return(x):
    p = np.prod(x+1)
    n = x.shape[0]
    return(p**(1/n*12)-1)*100

# 2. Get the Data
## 2.1. Industrial Production of the US

y_us = Fred('INDPRO', 'y_us_gap', plot=False)
y_us_k = Kalman_Filter(j=0.999995, p=.00050,v=0.1, x= y_us, roll= 10,sd=1,plot=False)
y_us_gap = y_us_k.iloc[:,2:3]
y_us_gap = y_us_gap.rename(columns={"y_us_gap_gap": "y_us_gap"})

## 2.2. Fed Funds of the US

i_us = Fred('FEDFUNDS', 'i_us_t-1', plot=False).shift(1).dropna()

## 2.3. Industrial Production of Mexico

y_mx = Banxico(serie="sr16734",name='y_mx_gap', plot=False)
y_mx_k = Kalman_Filter(x= y_mx,j=0.999995, p=.00050,v=0.1,roll=10,sd=1,plot=False)
y_mx_gap = y_mx_k.iloc[:,2:3]
y_mx_gap = y_mx_gap.rename(columns={"y_mx_gap_gap": "y_mx_gap"})

## 2.4. Interest Rate of Mexico

i_mx = Banxico(serie="SF283",name='i_mx_t-1', plot=False).shift().dropna() # TIIE 28 day

## 2.5. Inflation of US

inf_dev_us = Fred('CPIAUCSL', 'inf_dev_us', plot=False).pct_change(periods=12).dropna()-0.02

## 2.6. Inflation Mexico

inf_dev_mx = Banxico(serie="sp1",name='inf_dev_mx', plot=False).pct_change(periods=12).dropna()-0.03

## 2.7. Exchange Rate

er = Banxico(serie="SF17906",name='ER', plot=False)

##  2.8. Monthly Returns of Exchange Rate

er_pct_t0 = er.pct_change(periods=1).dropna()
er_pct_t0.rename(columns={'ER': 'ER_s' }, inplace=True)

er_pct_t1 = er_pct_t0.shift(1)
er_pct_t1.rename(columns={'ER_s': 'ER_s+1' }, inplace=True)

er_pct_t2 = er_pct_t0.shift(2)
er_pct_t2.rename(columns={'ER_s': 'ER_s+2' }, inplace=True)

er_pct_t3 = er_pct_t0.shift(3)
er_pct_t3.rename(columns={'ER_s': 'ER_s+3' }, inplace=True)

er_pct_t6 = er_pct_t0.shift(6)
er_pct_t6.rename(columns={'ER_s': 'ER_s+6' }, inplace=True)

er_pct_t12 = er_pct_t0.shift(12)
er_pct_t12.rename(columns={'ER_s': 'ER_s+12' }, inplace=True)

##  2.9. Dataset

df = pd.concat([er, er_pct_t0, er_pct_t1,er_pct_t2,er_pct_t3,
                er_pct_t6,er_pct_t12], axis=1).reindex(er_pct_t0.index)

df = pd.concat([df,y_mx_gap,y_us_gap,inf_dev_mx,inf_dev_us,i_mx,i_us], axis=1).reindex(df.index).dropna()
df.columns

## 2.10. Create Differentials

df['inf_diff'] = df['inf_dev_mx']-df['inf_dev_us']
df['i_diff'] = df['i_mx_t-1']-df['i_us_t-1']
df['y_diff'] = df['y_mx_gap']-df['y_us_gap']

## 2.11. Create empty columns for fitted values of changes i.e. changes of log exchange rates

df['fcast_ols'] = np.nan
df['fcast_rf'] = np.nan

## 2.12. Float numbers

df.columns
df.loc[:,'ER'] = df.loc[:,'ER'].apply(float)
df.loc[:,'ER_s'] = df.loc[:,'ER_s'].apply(float)
df.loc[:,'i_diff'] = df.loc[:,'i_diff'].apply(float)
df.loc[:,'inf_diff'] = df.loc[:,'inf_diff'].apply(float)
df.loc[:,'y_diff'] = df.loc[:,'y_diff'].apply(float)

# 3. Models
# 3.1. Papell Model

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(max_depth=2, random_state=0)

# Get numpy array
y = df['ER_s'].dropna().to_numpy().reshape(-1,1)
X = df[['i_diff', 'inf_diff','y_diff']].dropna().to_numpy()

# Random Forest
for i in range(120,len(df)):
    tmp = random_forest.fit(X[i-120:i],y[i-120:i])
    df.iloc[i:i+1,df.columns.get_loc('fcast_rf')] = tmp.predict(X[i:i+1])[0]

# OLS
for i in range(120,len(df)):
    tmp = smf.ols(formula = 'ER_s ~ inf_diff + y_diff + i_diff', data=df[i-120:i]).fit()
    df.iloc[i:i+1,df.columns.get_loc('fcast_ols')] = tmp.predict(df[['inf_diff', 'y_diff', 'i_diff']][i:i+1])[0]


df['profit_tr'] = 1000
df['profit_rw'] = 1000


for i in range(1,len(df.index)):
        if df['fcast_ols'][i] > 0:
            df['profit_tr'][i] = df['profit_tr'][i-1] * (1+df['ER_s'][i])
        elif df['fcast_ols'][i] < 0:
            df['profit_tr'][i] = df['profit_tr'][i-1] * (1+(df['ER_s'][i] * -1))


# Random Walk Strategy

for i in range(120,len(df.index)):
        if df['ER_s'][i-1] > 0:
            df['profit_rw'][i] = df['profit_rw'][i-1] * (1+df['ER_s'][i])
        elif df['ER_s'][i-1] < 0:
            df['profit_rw'][i] = df['profit_rw'][i-1] * (1+(df['ER_s'][i] * -1))
            

# 3.2. Kalman Filter on Interest Differential

i_mx_t = Banxico(serie="SF283",name='i_mx', plot=False).dropna() # TIIE 28 day
i_us_t = Fred('FEDFUNDS', 'i_us', plot=False).dropna()
i_diff = pd.DataFrame(i_mx_t['i_mx'] - i_us_t['i_us']).dropna()
i_diff = i_diff.rename(columns={0: 'i_diff'}) #rename
i_diff = i_diff.iloc[120:,:]
# Exporting results to csv
#i_diff.to_csv(r'i_diff_120121.csv', index = True)
# Kalman Filter Strategy Cumulative Optimization

def KF_opt(x= i_diff,k=0.12, roll=3, sd=0.25, n=1, plot=False):
    kf_model = KF(x= i_diff, k=k, roll=roll,sd=sd,plot=False)
    kf_model['er'] = er
    kf_model['return_er'] = kf_model['er'].pct_change(1)
    kf_model['returns'] = 1000
    kf_model['strategy'] = 'long'
    n = n

    if kf_model['i_diff_gap'][0] > 0:
        kf_model['strategy'][0]="long"
    else:
        kf_model['strategy'][0]="short"
    
    for i in range(1,len(kf_model.index)):
        if i % n == 0:
            if kf_model['strategy'][i-1] =='long' and kf_model['i_diff_gap'][i] > 0:
                kf_model['strategy'][i] = "long"
            elif kf_model['strategy'][i-1] =='long' and kf_model['i_diff_gap'][i] < 0 and abs(kf_model['i_diff_gap'][i]) > kf_model['i_diff_std_roll+'][i]:
                kf_model['strategy'][i] = "short"
            elif kf_model['strategy'][i-1] =='short' and kf_model['i_diff_gap'][i] < 0:
                kf_model['strategy'][i] = "short"
            elif kf_model['strategy'][i-1] =='short' and kf_model['i_diff_gap'][i] > 0 and kf_model['i_diff_gap'][i] > kf_model['i_diff_std_roll+'][i]:
                kf_model['strategy'][i] = "short"
        else:
            kf_model['strategy'][i] = kf_model['strategy'][i-1]
    
    for i in range(1,len(kf_model.index)):
        if kf_model['strategy'][i-1] == 'short':
            kf_model['returns'][i] = kf_model['returns'][i-1] * (1+kf_model['return_er'][i])
        elif kf_model['strategy'][i-1] == 'long':
            kf_model['returns'][i] = kf_model['returns'][i-1] * (1+(kf_model['return_er'][i]*-1))
    
       
    plt.figure(figsize=(26,8))
    plt.style.use('ggplot')
    line, = plt.plot(kf_model.index,kf_model['returns'], lw=2, linestyle='-', color='b')
    plt.gca().set(title='Cumulative Profit', xlabel = 'Date', ylabel = 'variable')
    plt.xticks(np.arange(0, len(kf_model['returns']), step=24), rotation=90)
    plt.show()

    return kf_model['returns'][-1], kf_model['returns'], kf_model['strategy']


# Optimization

#The first step is to create the vectors of hyper parameters to modify

k = np.arange(0.0, 1, 0.01).tolist()
sd = np.arange(0.0, 2.5, 0.5).tolist()
n = np.arange(0.0, 3, 1).tolist()


#Then we will crate a function that makes combinations of all the parameters in a 3D matrix

def comb_elements(list_a,list_b,list_c):
    combination = []
    for i in list_a:
        for j in list_b:
            for k in list_c:
                combination.append([i,j,k])
    return pd.DataFrame(combination)

#This is useful because we test a lot of combinations of hyper parameters with just a few lines of code.

combinations = comb_elements(list_a=k, list_b=sd,list_c=n) #combination of the hyper parameters

profit = []

#for i in range(0,len(combinations)):
#    df1 = KF_opt(x= i_diff,k=combinations[0][i], roll=2, sd=combinations[1][i], n=combinations[2][i], plot=False)
#    profit.append(df1[0])

#optimization = pd.DataFrame(profit, columns={'profit'})
#optimization = optimization.sort_values(['profit'], ascending = (False))

# Best kalman gain
#combinations.loc[optimization.index[0:10]]


# Get the best kalman filter model
df2 = KF_opt(x= i_diff,k=0.13, roll=2, sd=1, n=1, plot=False)
df3 = pd.DataFrame(df2[2])
df2 = pd.DataFrame(df2[1])

# 4. Evaluate Portfolio

# 4.1. Sharpe Ratio
monthly_risk_free_rate = (1+i_mx_t.iloc[120:,:]/100)**(1/12) - 1
df['returns_tr'] = df['profit_tr'].pct_change(1)
df['returns_rw'] = df['profit_rw'].pct_change(1)
df = df.iloc[119:,:]
df['profit_kf'] = df2 
df['returns_kf'] = df['profit_kf'].pct_change(1)
df = df.dropna()

# 4.0. Geometric Average Return Rate

print("The geometric average return of Kalman Filter is ", geo_return(x=df['returns_kf']))
print("The geometric average return of Taylor Rule is ", geo_return(x=df['returns_tr']))
print("The geometric average return of Random Walk is ", geo_return(x=df['returns_rw']))

# 4.1. Sharpe Ratio
print("The sharpe ratio for kalman filter is ", sharpe_ratio(df['returns_kf'].dropna(), monthly_risk_free_rate['i_mx']))
print("The sharpe ratio for taylor rule| is ", sharpe_ratio(df['returns_tr'].dropna(), monthly_risk_free_rate['i_mx']))
print("The sharpe ratio for random walk is ", sharpe_ratio(df['returns_rw'].dropna(), monthly_risk_free_rate['i_mx']))


# 4.2. Calculate 5% value at risk

print("The VaR 5% for kalman filter is ",np.quantile(df['returns_kf'].dropna(), 0.05))
print("The VaR 5% for taylor rule is ",np.quantile(df['returns_tr'].dropna(), 0.05))
print("The VaR 5% for random walk is ",np.quantile(df['returns_rw'].dropna(), 0.05))

# 4.3. Plot Returns Distribution

title = 'Kalman Filter Distribution of Returns'
plot_returns(df, 'returns_kf', title)
title = "Papell's Model Distribution of Returns"
plot_returns(df, 'returns_tr', title)
title = 'Random Walk Distribution of Returns'
plot_returns(df, 'returns_rw', title)

# 4.4. Skewness

print( "The skewness for kalman filter is ",skewness(df['returns_kf']))
print( "The skewness for taylor rule is ",skewness(df['returns_tr']))
print( "The skewness for random walk is ",skewness(df['returns_rw']))

# 4.5. Gini index

print("The gini index for kalman filter is ", Gini_index(x=df['returns_kf']))
print("The gini index for taylor rule is ", Gini_index(x=df['returns_tr']))
print("The gini index for random walk is ", Gini_index(x=df['returns_rw']))

###############################################################################
# 5. Evaluate forecasts
###############################################################################

# If forecast depreciation, then D=1. Forcast appreciation, then D=-1
# Kalman filter strategy
df['D_kf'] = 0
df.loc[(df3['strategy'] == 'short'),'D_kf'] = 1 #peso depreciation
df.loc[(df3['strategy'] == 'long'),'D_kf'] = -1 # peso appreciation

# Taylor Rule strategy
df['D_tr'] = 0
df.loc[(df['fcast_ols'] > 0),'D_tr'] = 1 #peso depreciation
df.loc[(df['fcast_ols'] == 0),'D_tr'] = 0
df.loc[(df['fcast_ols'] < 0),'D_tr'] = -1 # peso appreciation


# Random Walk strategy
df['D_rw'] = 0
df.loc[(df['ER_s+1'] > 0),'D_rw'] = 1 #peso depreciation
df.loc[(df['ER_s+1'] == 0),'D_rw'] = 0
df.loc[(df['ER_s+1'] < 0),'D_rw'] = -1 # peso appreciation


# Let's look at the distribution
df.D_kf.value_counts()/df.shape[0]
df.D_tr.value_counts()/df.shape[0]
df.D_rw.value_counts()/df.shape[0]

# Realized Directional Change
df['R'] = 0
df.loc[(df['ER_s'] > 0),'R'] = 1
df.loc[(df['ER_s'] == 0),'R'] = 0
df.loc[(df['ER_s'] < 0),'R'] = -1

# Let's look at the distribution

df.R.value_counts()/df.shape[0]

## Success Ratio Kalman Filter
df['S_kf'] = 0
df.loc[(df['D_kf']==df['R']), 'S_kf'] = 1
success_ratio_kf = np.sum(df['S_kf'])/df.shape[0]
print('Success Ratio Kalman Filter : ', success_ratio_kf)

## Success Ratio Taylor Rule
df['S_tr'] = 0
df.loc[(df['D_tr']==df['R']), 'S_tr'] = 1
success_ratio_tr = np.sum(df['S_tr'])/df.shape[0]
print('Success Ratio Taylor Rule : ', success_ratio_tr)

## Success Ratio Taylor Rule
df['S_rw'] = 0
df.loc[(df['D_rw']==df['R']), 'S_rw'] = 1
success_ratio_rw = np.sum(df['S_rw'])/df.shape[0]
print('Success Ratio Taylor Rule : ', success_ratio_rw)

## Sample Covariance
def binomial(d_fcast):
    dfb = df.copy()
    dfb['W'] = (dfb[d_fcast]-np.mean(dfb[d_fcast]))*(dfb['R']-np.mean(dfb['R']))
    T_B = np.mean(dfb['W'])
    print('The statistic is: ', T_B)

    ## Newey-West LRV estimator

    dy = dfb['W'] - np.mean(dfb['W'])
    gamma_0 = sum((dy)**2)/len(dfb)
    gamma_1 = np.mean((dy*dy.shift(-1))[:len(dfb)-1])
    LRV = gamma_0 + 2 * (1-1/2) * gamma_1
    ## Test-statistic


    statistic = T_B/np.sqrt(LRV/dfb.shape[0])
    print('The standard error is  : ', np.sqrt(LRV/dfb.shape[0]))
    print('Test statistic : ', statistic, ', 5 % critical value : ', round(norm.ppf(0.95),2))
    print('p-value:  : ', scipy.stats.norm.sf(abs(statistic)))


binomial(d_fcast='D_kf')
binomial(d_fcast='D_tr')
binomial(d_fcast='D_rw')


# Weighted Mean
#df['W_2'] = df['D']*(df['s_future']-df['s_current'])
def wm(d_fcast):
    dfa = df.copy()
    dfa['W_2'] = dfa[d_fcast]*(dfa['ER']-dfa['ER'].shift(1))
    dfa = dfa.dropna()
    T_WB = np.mean(dfa['W_2'])
    print('The statistic is: ', T_WB)

    ## Newey-West LRV estimator
    dy_2 = dfa['W_2'] - np.mean(dfa['W_2'])
    gamma_0 = sum((dy_2)**2)/len(dfa)
    gamma_1 = np.mean((dy_2*dy_2.shift(-1))[:len(dfa)-1])
    LRV_2 = gamma_0 + 2 * (1-1/2) * gamma_1

    ## Test-statistic
    statistic_2 = T_WB/np.sqrt(LRV_2/len(dfa))
    print('The standard error is  : ', np.sqrt(LRV_2/len(dfa)))
    print('Test statistic : ', statistic_2, ', 5 % critical value : ', round(norm.ppf(0.95),2))
    print('p-value:  : ', scipy.stats.norm.sf(abs(statistic_2)))

wm(d_fcast='D_kf')
wm(d_fcast='D_tr')
wm(d_fcast='D_rw')


# Exporting results to csv
#df.to_csv(r'Results_120121.csv', index = True)

