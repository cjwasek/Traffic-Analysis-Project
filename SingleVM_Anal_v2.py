#import socket
import sys
#import time
import math
import statistics as stats
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt



'''This function collects timestamps from a .txt file and converts them from a
string to a list of floats'''
def time_collect():
    
    vm_times = []
    host_times = []
    
    run = f.readline()
    vm1_times = f.readline().split()
    for i in range(len(vm1_times)):
        if i == 0:
            x = vm1_times[i]
            x = x[1:-1]
            vm1_times[i] = float(x)
        else:
            x = vm1_times[i]
            x = x[:-1]
            vm1_times[i] = float(x)
    vm_times.append(vm1_times)
    
    hostvm1_times = f.readline().split()
    for i in range(len(hostvm1_times)):
        if i == 0:
            x = hostvm1_times[i]
            x = x[1:-1]
            hostvm1_times[i] = float(x)
        else:
            x = hostvm1_times[i]
            x = x[:-1]
            hostvm1_times[i] = float(x)
    host_times.append(hostvm1_times)        
        
    return vm_times, host_times    
   


'''This function is designed to calculate the frequency of the targeted system's 
OS given the values of corresponding TCP packet timestamps and host system clock
measurments.  The calculated frequency is then rounded to the nearest 'TRUE' 
frequency (e.g. 1 Hz, 100 Hz, 250 Hz, etc.).'''

def calc_freq(s, t):
    
    freq_true = (t[-1] - t[0]) / (s[-1] - s[0])
    
    if freq_true <= 5:
        freq = 1
    elif freq_true <= 50:
        freq = 10
    elif freq_true <= 175:
        freq = 100
    elif freq_true <= 375:
        freq = 250
    elif freq_true <= 750:
        freq = 500
    else:
        freq = 1000
    
    return freq


'''This function is designed to calculate the elapsed time of the targeted VM 
given TCP timestamps and frequency of the system's OS.'''

def target_time(t, f):
    w = []
    for i in range(len(t)):
        w.append((t[i] - t[0]) / f)    
    return w


'''This function is designed to calculate the elapsed time of the host given 
system clock measurements.'''

def host_time(s):
    x = []
    for i in range(len(s)):
        x.append(s[i] - s[0])
    return x


'''This function is designed to calculate the degree of time offset of the 
targeted VM relative to the host.'''
    
def offset(w, x):
    y = []
    for i in range(len(w)):
        y.append(w[i] - x[i])
    return y


'''This calculates the t-value for the VM.'''
def model_analysis(x, x_matrix, y, line, y_hat, b):
    n = len(x) # number of samples
    s_x = stats.stdev(x) # standard deviation of x values
    s_y = stats.stdev(y) # standard deviation of y values
    s2_x = stats.variance(x) # variance of x values
    s2_y = stats.variance(y) # variance of y values
    s_xy = b * s2_x # covariance of VM
    
    mad_temp = 0
    SSE = 0
    for i in range(len(y)):
        temp = abs(y[i] - y_hat[i])
        mad_temp += temp
        SSE += temp**2 # sum of squares for error
    MAD = mad_temp / n    
    s_err = math.sqrt(SSE / (n - 2)) # standard error of estimate
    s_b = s_err / math.sqrt((n - 1) * s2_x)
    
    r = s_xy / (s_x * s_y) # sample coefficient of correlation
    R_2 = line.score(x_matrix, y) # coefficient of determination 
    R_2calc = s_xy**2 / (s2_x * s2_y)
    t = b / s_b # t-value for slope assuming true slope = 0
    
    f1.write('\nSkew = ' + str(b) + '\n')
    f1.write('Coefficient of correlation (r) = ' + str(r) + '\n')
    #f1.write('Coefficient of determination (R^2) via scikit = ' + str(R_2) + '\n')
    f1.write('Coefficient of determination (R^2) calculate = ' + str(R_2calc) + '\n')
    f1.write('Test statistic for clock skew (t) = ' + str(t) + '\n')
    f1.write('Mean Absolute Deviation (MAD) = ' + str(MAD) + '\n')
    f1.write('Sum of Squares for Forecast Error (SSE) = ' + str(SSE) + '\n')
    
    return


'''This function is designed to run the initial clock skew analysis.'''
def skew_analysis(tcp_time, sys_time):
    
    x = []
    y = []
        
    for i in range(len(tcp_time)):
        freq = calc_freq(sys_time[i], tcp_time[i]) # frequency (F)
        tgt_timelapse = target_time(tcp_time[i], freq) # elapsed time of target (w) ((T_1 - T_0)/F)
        host_timelapse = host_time(sys_time[i]) # elapsed time of host (x) (t_1 - t_0)
        tgt_offset = offset(tgt_timelapse, host_timelapse) # observed offset of VM (y) (w - x)
    
        x.append(host_timelapse)
        y.append(tgt_offset)
                
    return (x, y)

'''This function returns a portion of the total timestamp list given the full
list of x and y values, the percentage of the list you want returned (p), and 
which percentage piece you want back(n) ex. p = 10 and n = 3 means I want the 
list divided into 10 equal parts and I want the 3rd part back'''
def time_splice(x, y, p, n):
    
    print(len(x))
    perc = int(len(x) / p)
    print('perc = ', perc)
    start = perc * (n - 1)
    print('start = ', start)
    end = start + perc
    print('end = ',end )
    x_ret = x[start:end]
    y_ret = y[start:end]
    
    print('x_ret is: ', x_ret)
    
    return x_ret, y_ret


'''This function generates points to plot the maximum plot values '''
def wave_rider(sys_time, tcp_time):
    
    x_temp = sys_time
    y_temp = tcp_time
    
    temp_x = []
    temp_y = []
    
    for j in range(4):
        
        for i in range(len(y_temp) -2):
            if y_temp[i] == y_temp[0]:
                if y_temp[i] > y_temp[i+1]:
                    temp_x.append(x_temp[i])
                    temp_y.append(y_temp[i])
            elif y_temp[i] == y_temp[-3]:
                if (y_temp[i+1] > y_temp[i]) and (y_temp[i+1] > y_temp[i+2]):
                    temp_x.append(x_temp[i+1])
                    temp_y.append(y_temp[i+1])
                elif y_temp[-1] > y_temp[1+1]:
                    temp_x.append(x_temp[-1])
                    temp_y.append(y_temp[-1])                    
            elif (y_temp[i+1] > y_temp[i]) and (y_temp[i+1] > y_temp[i+2]):
                temp_x.append(x_temp[i+1])
                temp_y.append(y_temp[i+1])
        x_temp = temp_x
        y_temp = temp_y
        
        temp_x = []
        temp_y = []
        
    x = x_temp
    y = y_temp
    
    return (x, y)
        


'''This function graphs scatter plots and clock skew estimates '''
def plot_data(x_vals, y_vals):
    
    trans_vals1 = x_vals
    x_vm1 = np.matrix(x_vals).T
    y_vm1 = y_vals
    #trans_vals1 = x_vals[0]
    #x_vm1 = np.matrix(x_vals[0]).T
    #y_vm1 = y_vals[0]
    
     
    # Set up linear regression, Theil-Sen regression, and RANSAC models
    lin_vm1 = linear_model.LinearRegression()
    lin_vm1.fit(x_vm1, y_vm1)
    #ts_vm1 = linear_model.TheilSenRegressor()
    #ts_vm1.fit(x_vm1, y_vm1)
    ransac_vm1 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.01)
    ransac_vm1.fit(x_vm1, y_vm1)
    

    # Define RANSAC inliers and outliers  
    inliers_vm1 = ransac_vm1.inlier_mask_
    outliers_vm1 = np.logical_not(inliers_vm1)
    
    vm1_xinliers = []
    vm1_yinliers = []
    vm1_xoutliers = []
    vm1_youtliers = []
    for i in range(len(trans_vals1)):
        if inliers_vm1[i] == True:
            vm1_xinliers.append(trans_vals1[i])
            vm1_yinliers.append(y_vm1[i])
        else:
            vm1_xoutliers.append(trans_vals1[i])
            vm1_youtliers.append(y_vm1[i])  
    vm1t_xinliers = np.matrix(vm1_xinliers).T
    #print('The VM1 inliers are: ', inliers_vm1)
    #print('The VM1 outliers are: ', outliers_vm1)
        
    
    # Create plot lines for regression models
    linreg_vm1 = lin_vm1.predict(x_vm1)
    #tsreg_vm1 = ts_vm1.predict(x_vm1)
    ranreg_vm1 = ransac_vm1.predict(x_vm1)
        
    
    # Determine skew values for regression models
    linskew_vm1 = lin_vm1.coef_
    #tsenskew_vm1 = ts_vm1.coef_
    ransacskew_vm1 = ransac_vm1.estimator_.coef_
    #print(ransac_vm1.get_params())
    
    
    # Create plot model for max values
    
    x_wav, y_wav = wave_rider(x_vals, y_vals)
    #x_wav, y_wav = wave_rider(x_vals[0], y_vals[0])
    xt_wav = np.matrix(x_wav).T
    lin_wav = linear_model.LinearRegression()
    lin_wav.fit(xt_wav, y_wav)
    linreg_wav = lin_wav.predict(xt_wav)
    linskew_wav = lin_wav.coef_
    
        
    
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nThe data analysis for each regression model on VM1 is as follows:')
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nLinear Regression Model')
    model_analysis(x_vals, x_vm1, y_vm1, lin_vm1, linreg_vm1, linskew_vm1)
    #model_analysis(x_vals[0], x_vm1, y_vm1, lin_vm1, linreg_vm1, linskew_vm1)
    #f1.write('\nTheil-Sen Regression Model')
   # model_analysis(x_vals[0], x_vm1, y_vm1, ts_vm1, tsreg_vm1, tsenskew_vm1)
    f1.write('\nRANSAC Regression Model')
    model_analysis(vm1_xinliers, vm1t_xinliers, vm1_yinliers, ransac_vm1, ranreg_vm1, ransacskew_vm1)
    f1.write('\nWave Rider Model')
    model_analysis(x_wav, xt_wav, y_wav, lin_wav, linreg_wav, linskew_wav)    
    
    fig, vm1 = plt.subplots()
    #fig, (vm1, movavg) = plt.subplots(2)
    #fig, ((vm1, vm2), (vm3, vm4)) = plt.subplots(2, 2, sharex = 'col', sharey = 'row')
    #fig.title('VM Skew Analysis')
    #fig.xlabel('Host System Time Lapse in seconds (t_i - t_0)')
    #fig.ylabel('Observed VM Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    
    #plt.subplot(2, 1, 1)
    #vm1.plot(x_vm1, y_vm1, '.k')
    vm1.plot(vm1_xinliers, vm1_yinliers, '.k')
    vm1.plot(vm1_xoutliers, vm1_youtliers, '.r') 
    vm1.plot(x_wav, y_wav, '.c')
    vm1.plot(x_vm1, linreg_vm1, '-m', label = 'Lin Reg Skew = %s' % linskew_vm1)
   #vm1.plot(x_vm1, tsreg_vm1, '-b', label = 'TS Reg Skew = %s' % tsenskew_vm1)
    vm1.plot(x_vm1, ranreg_vm1, '-g', label = 'RANSAC Reg Skew = %s' % ransacskew_vm1)
    vm1.plot(xt_wav, linreg_wav, '-r', label = 'Wave Rider Skew = %s' % linskew_wav)
    #vm1.plot([x0_vm1, x1_vm1], [y0_vm1, y1_vm1], c='r')
    #vm1.plot([tsx0_vm1, tsx1_vm1], [tsy0_vm1, tsy1_vm1], c='g')
    vm1.set_title('Pcap Splice Analysis')
    vm1.set_ylabel('Observed VM Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    #vm1.axis([-5, 15, -0.005, 0.005])
    vm1.axis([0, 650, -0.03, 0.03])
    vm1.legend(loc = 'upper right')
    vm1.text(20, 0.02, r'# Inliers = %s' % len(vm1_xinliers)) 
    
    #plt.subplot(2, 1, 1)
    #vm1.plot(x_vm1, y_vm1, '.k')
    #movavg.plot(trans_vals1, y_vm1, '-k')
    #movavg.plot(xt_wav, linreg_wav, '-r')
    #vm1.plot([x0_vm1, x1_vm1], [y0_vm1, y1_vm1], c='r')
    #vm1.plot([tsx0_vm1, tsx1_vm1], [tsy0_vm1, tsy1_vm1], c='g')
    #movavg.set_title('Moving Average Analysis')
    #movavg.set_ylabel('Observed VM Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    #vm1.axis([-5, 15, -0.005, 0.005])
    #movavg.axis([0, 650, -0.03, 0.03])
    #movavg.legend(loc = 'upper right')
    #movavg.text(20, 0.02, r'# Inliers = %s' % len(vm1_xinliers))    
        
    plt.show()
    
    return


#*************************
# ***** MAIN PROGRAM *****
#*************************

fn = 'Pcap_Parsed_Times.txt'
fn1 = 'Pcap_Splice_Analysis_0.txt'

f = open(fn, 'r')
f1 = open(fn1, 'w')    

vm_timestamps = []
host_timestamps = []
x_values = []
y_values = []
    
vm_timestamps, host_timestamps = time_collect()

#f1.write('***********\n')
#f1.write('Test Run #1\n')
#f1.write('***********\n')
x_values, y_values = skew_analysis(vm_timestamps, host_timestamps)

#x_mini, y_mini = time_splice(x_values[0], y_values[0], 10, 1)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 10, 3)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 10, 5)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 10, 7)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 10, 10)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 5, 1)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 5, 3)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 5, 4)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 2, 1)
#x_mini, y_mini = time_splice(x_values[0], y_values[0], 2, 2)

#print(type(x_mini))
#print(x_mini)


plot_data(x_values[0], y_values[0])
#plot_data(x_mini, y_mini)

f.close()
f1.close()
