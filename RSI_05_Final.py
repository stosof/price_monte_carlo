import csv
import numpy as np
import talib as ta
import math
import plotly as py
import plotly.graph_objs as go
from plotly import tools
import statsmodels.api as sm 
import scipy.interpolate as interpolate
import scipy.stats as ss
import time 
from random import gauss

divide_into_minutes = 60
def transform_minute_to_hour(array):
    
    minute = 0
    array_hour = []
    append_tmp = []
    open_tmp = 0
    high_tmp = 0
    low_tmp = 100
    close_tmp = 0
    
    for i in range(len(array)):
        
        if minute == divide_into_minutes:
            close_tmp = array[i][3]
            append_tmp.append(open_tmp)
            append_tmp.append(high_tmp)
            append_tmp.append(low_tmp)
            append_tmp.append(close_tmp)
            array_hour = array_hour + [append_tmp]
            minute = 0
            append_tmp = []
            open_tmp = 0
            high_tmp = 0
            low_tmp = 100
            close_tmp = 0
            continue
        
        if minute == 0:
            open_tmp = array[i][0]
        
        if float(array[i][1]) > float(high_tmp):
            high_tmp = array[i][1]

        if float(array[i][2]) < float(low_tmp):

            low_tmp = array[i][2]
        
        minute += 1
    return array_hour

with open('DAT_ASCII_EURUSD_M1_20155.csv', 'rb') as f:
    reader = csv.reader(f)
    data_0 = list(reader)

data = transform_minute_to_hour(data_0)

close = np.zeros(shape=(len(data)))
for i in range(len(data)):
#    
    close[i] = float(data[i][3])

starting_asset_price = float(close[len(close)-1])

output = ta.SMA(close,5)
output_1 = ta.SMA(close,15)
output_2 = ta.RSI(close,7)
output_3 = ta.RSI(close,14)

log_return = []

for i in range(len(data)-1):
    
    return_tmp = math.log((float(data[i+1][3])/float(data[i][3])))
    log_return.append(return_tmp)

rsi_buckets = []

for i in range(10):

    rsi_buckets.append([])

for i in range(len(data)):
#    
    data[i].append(output_2[i])
#
for i in range(len(log_return)):

    if math.isnan(data[i][4]):
        continue
#    
    bucket_tmp = math.trunc((data[i][4]/10))
#    
    rsi_buckets[bucket_tmp].append(log_return[i])

sample = rsi_buckets[8]
ecdf = sm.distributions.ECDF(sample)

x_sample = np.linspace(min(sample), max(sample),100)
y_sample = ecdf(x_sample)
 
def inverse_transform_sampling(data_in, n_bins, rand_num):
    hist, bin_edges = np.histogram(data_in, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    return inv_cdf(rand_num)

def call_payoff(S_T,K):
    return max(0.0,float(S_T)-float(K))


num_simulations = 1000
steps_to_simulate = 240
data_predicted_all = []
for i in range(num_simulations):
    last_data = data[len(data)-1]
    data_predicted = []
    close_tmp = close[-25:]
    for s in range(steps_to_simulate):
        r = np.random.rand(1)
        curr_rsi_bucket = math.trunc(last_data[4]/10)
        q_sample = inverse_transform_sampling(rsi_buckets[curr_rsi_bucket], 1, r)
        
        new_close = float(last_data[3]) * math.exp(float(q_sample))
        close_tmp = np.append(close_tmp,new_close)
        rsi_updated = ta.RSI(close_tmp,7)
        
        data_tmp = [None,None,None, close_tmp[len(close_tmp)-1],rsi_updated[len(rsi_updated)-1]]
        data_predicted.append(data_tmp)
        last_data = data_tmp
    
    data_predicted_all.append(data_predicted)

call_payoff_sum = []
r = 0.0014 # rate of 0.14%
T = ((steps_to_simulate * divide_into_minutes)/1440) / 365.0
discount_factor = math.exp(-r * T)
option_strike_price = 1.09

fig = tools.make_subplots(rows=1,cols=1)
last_data_1 = data[len(data)-1]

for i in range(len(data_predicted_all)):
    sim_close = []
    sim_close.append(last_data_1[3])
    for s in range(len(data_predicted_all[i])):
        sim_close.append(data_predicted_all[i][s][3])
    call_payoff_sum.append( call_payoff(float(sim_close[len(sim_close)-1]),option_strike_price) )
    trace_tmp = go.Scatter(y = sim_close)
    fig.append_trace(trace_tmp,1,1)

py.offline.plot(fig)

price = float(discount_factor) * (sum(call_payoff_sum) / float(len(data_predicted_all)))
print 'RSI price estimation: ', price
    

# _____________________________________ ANALYTICAL ______________________________

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

S0 = starting_asset_price
K = option_strike_price
r = 0.0014
sigma = 0.1587
T = ((steps_to_simulate * divide_into_minutes)/1440) / 365.0
Otype='C'


print "S0\tstock price at time 0:", S0
print "K\tstrike price:", K
print "r\tcontinuously compounded risk-free rate:", r
print "sigma\tvolatility of the stock price per year:", sigma
print "T\ttime to maturity in trading years:", T


t=time.time()
c_BS = BlackScholes(Otype,S0, K, r, sigma, T)
elapsed=time.time()-t
print "c_BS\tBlack-Scholes price:", c_BS, elapsed


#_____________________________ STANDARD MONTE CARLO ___________________________


def generate_asset_price(S,v,r,T):
    return S * math.exp((r - 0.5 * v**2) * T + v * math.sqrt(T) * gauss(0,1.0))

def call_payoff(S_T,K):
    return max(0.0,S_T-K)

S = starting_asset_price # underlying price
v = 0.1587
r = 0.0014 # rate of 0.14%
T = ((steps_to_simulate * divide_into_minutes)/1440) / 365.0
K = option_strike_price
simulations = 1000000
payoffs = []

for i in xrange(simulations):
    S_T = generate_asset_price(S,v,r,T)
    payoffs.append(
        call_payoff(S_T, K)
    )

price = discount_factor * (sum(payoffs) / float(simulations))
print 'Standard Monte Carlo Price: %.6f' % price




