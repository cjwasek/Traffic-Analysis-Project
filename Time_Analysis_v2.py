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
    
    vm2_times = f.readline().split()
    for i in range(len(vm2_times)):
        if i == 0:
            x = vm2_times[i]
            x = x[1:-1]
            vm2_times[i] = float(x)
        else:
            x = vm2_times[i]
            x = x[:-1]
            vm2_times[i] = float(x)
    vm_times.append(vm2_times)
    
    hostvm2_times = f.readline().split()
    for i in range(len(hostvm2_times)):
        if i == 0:
            x = hostvm2_times[i]
            x = x[1:-1]
            hostvm2_times[i] = float(x)
        else:
            x = hostvm2_times[i]
            x = x[:-1]
            hostvm2_times[i] = float(x)
    host_times.append(hostvm2_times)  
    
    vm3_times = f.readline().split()
    for i in range(len(vm3_times)):
        if i == 0:
            x = vm3_times[i]
            x = x[1:-1]
            vm3_times[i] = float(x)
        else:
            x = vm3_times[i]
            x = x[:-1]
            vm3_times[i] = float(x)
    vm_times.append(vm3_times)
    
    hostvm3_times = f.readline().split()
    for i in range(len(hostvm3_times)):
        if i == 0:
            x = hostvm3_times[i]
            x = x[1:-1]
            hostvm3_times[i] = float(x)
        else:
            x = hostvm3_times[i]
            x = x[:-1]
            hostvm3_times[i] = float(x)
    host_times.append(hostvm3_times)
        
    vm4_times = f.readline().split()
    for i in range(len(vm4_times)):
        if i == 0:
            x = vm4_times[i]
            x = x[1:-1]
            vm4_times[i] = float(x)
        else:
            x = vm4_times[i]
            x = x[:-1]
            vm4_times[i] = float(x)
    vm_times.append(vm4_times)
    
    hostvm4_times = f.readline().split()
    for i in range(len(hostvm4_times)):
        if i == 0:
            x = hostvm4_times[i]
            x = x[1:-1]
            hostvm4_times[i] = float(x)
        else:
            x = hostvm4_times[i]
            x = x[:-1]
            hostvm4_times[i] = float(x)
    host_times.append(hostvm4_times)        
        
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
    
    SSE = 0
    for i in range(len(y)):
        SSE += (y[i] - y_hat[i])**2 # sum of squares for error
    s_err = math.sqrt(SSE / (n - 2)) # standard error of estimate
    s_b = s_err / math.sqrt((n - 1) * s2_x)
    
    r = s_xy / (s_x * s_y) # sample coefficient of correlation
    R_2 = line.score(x_matrix, y) # coefficient of determination 
    R_2calc = s_xy**2 / (s2_x * s2_y)
    t = b / s_b # t-value for slope assuming true slope = 0
    
    f1.write('\nSkew = ' + str(b) + '\n')
    f1.write('Coefficient of correlation (r) = ' + str(r) + '\n')
    f1.write('Coefficient of determination (R^2) via scikit = ' + str(R_2) + '\n')
    f1.write('Coefficient of determination (R^2) calculate = ' + str(R_2calc) + '\n')
    f1.write('Test statistic for clock skew (t) = ' + str(t) + '\n')
    
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

'''This function graphs scatter plots and clock skew estimates '''
def plot_data(x_vals, y_vals):
    
    trans_vals1 = x_vals[0]
    x_vm1 = np.matrix(x_vals[0]).T
    y_vm1 = y_vals[0]
    
    trans_vals2 = x_vals[1]
    x_vm2 = np.matrix(x_vals[1]).T
    y_vm2 = y_vals[1]
    
    trans_vals3 = x_vals[2]
    x_vm3 = np.matrix(x_vals[2]).T
    y_vm3 = y_vals[2]
        
    trans_vals4 = x_vals[3]
    x_vm4 = np.matrix(x_vals[3]).T
    y_vm4 = y_vals[3]    
  
    # Set up linear regression, Theil-Sen regression, and RANSAC models
    lin_vm1 = linear_model.LinearRegression()
    lin_vm1.fit(x_vm1, y_vm1)
    ts_vm1 = linear_model.TheilSenRegressor()
    ts_vm1.fit(x_vm1, y_vm1)
    ransac_vm1 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.015)
    ransac_vm1.fit(x_vm1, y_vm1)
    
    lin_vm2 = linear_model.LinearRegression()
    lin_vm2.fit(x_vm2, y_vm2)
    ts_vm2 = linear_model.TheilSenRegressor()
    ts_vm2.fit(x_vm2, y_vm2)
    ransac_vm2 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.015)
    ransac_vm2.fit(x_vm2, y_vm2)
    
    lin_vm3 = linear_model.LinearRegression()
    lin_vm3.fit(x_vm3, y_vm3)
    ts_vm3 = linear_model.TheilSenRegressor()
    ts_vm3.fit(x_vm3, y_vm3)
    ransac_vm3 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.015)
    ransac_vm3.fit(x_vm3, y_vm3)
    
    lin_vm4 = linear_model.LinearRegression()
    lin_vm4.fit(x_vm4, y_vm4)
    ts_vm4 = linear_model.TheilSenRegressor()
    ts_vm4.fit(x_vm4, y_vm4)
    ransac_vm4 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.015)
    ransac_vm4.fit(x_vm4, y_vm4)    
    
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
    #print('The VM1 inliers are: ', inliers_vm1)
    #print('The VM1 outliers are: ', outliers_vm1)
    
    inliers_vm2 = ransac_vm2.inlier_mask_
    outliers_vm2 = np.logical_not(inliers_vm2)
    
    vm2_xinliers = []
    vm2_yinliers = []
    vm2_xoutliers = []
    vm2_youtliers = []
    for i in range(len(trans_vals2)):
        if inliers_vm2[i] == True:
            vm2_xinliers.append(trans_vals2[i])
            vm2_yinliers.append(y_vm2[i])
        else:
            vm2_xoutliers.append(trans_vals2[i])
            vm2_youtliers.append(y_vm2[i])            
    #print('The VM2 inliers are: ', inliers_vm2)
    #print('The VM2 outliers are: ', outliers_vm2)   
    
    inliers_vm3 = ransac_vm3.inlier_mask_
    outliers_vm3 = np.logical_not(inliers_vm3)
    
    vm3_xinliers = []
    vm3_yinliers = []
    vm3_xoutliers = []
    vm3_youtliers = []
    for i in range(len(trans_vals3)):
        if inliers_vm3[i] == True:
            vm3_xinliers.append(trans_vals3[i])
            vm3_yinliers.append(y_vm3[i])
        else:
            vm3_xoutliers.append(trans_vals3[i])
            vm3_youtliers.append(y_vm3[i])            
    #print('The VM3 inliers are: ', inliers_vm3)
    #print('The VM3 outliers are: ', outliers_vm3)
    
    inliers_vm4 = ransac_vm4.inlier_mask_
    outliers_vm4 = np.logical_not(inliers_vm4)
    
    vm4_xinliers = []
    vm4_yinliers = []
    vm4_xoutliers = []
    vm4_youtliers = []
    for i in range(len(trans_vals4)):
        if inliers_vm4[i] == True:
            vm4_xinliers.append(trans_vals4[i])
            vm4_yinliers.append(y_vm4[i])
        else:
            vm4_xoutliers.append(trans_vals4[i])
            vm4_youtliers.append(y_vm4[i])            
    #print('The VM4 inliers are: ', inliers_vm4)
    #print('The VM4 outliers are: ', outliers_vm4)    
    
    # Create plot lines for regression models
    linreg_vm1 = lin_vm1.predict(x_vm1)
    tsreg_vm1 = ts_vm1.predict(x_vm1)
    ranreg_vm1 = ransac_vm1.predict(x_vm1)
    
    linreg_vm2 = lin_vm2.predict(x_vm2)
    tsreg_vm2 = ts_vm2.predict(x_vm2)
    ranreg_vm2 = ransac_vm2.predict(x_vm2) 
    
    linreg_vm3 = lin_vm3.predict(x_vm3)
    tsreg_vm3 = ts_vm3.predict(x_vm3)
    ranreg_vm3 = ransac_vm3.predict(x_vm3)
    
    linreg_vm4 = lin_vm4.predict(x_vm4)
    tsreg_vm4 = ts_vm4.predict(x_vm4)
    ranreg_vm4 = ransac_vm4.predict(x_vm4)    
    
    # Determine skew values for regression models
    linskew_vm1 = lin_vm1.coef_
    tsenskew_vm1 = ts_vm1.coef_
    ransacskew_vm1 = ransac_vm1.estimator_.coef_
    print(ransac_vm1.get_params())
    
    linskew_vm2 = lin_vm2.coef_
    tsenskew_vm2 = ts_vm2.coef_
    ransacskew_vm2 = ransac_vm2.estimator_.coef_
    
    linskew_vm3 = lin_vm3.coef_
    tsenskew_vm3 = ts_vm3.coef_
    ransacskew_vm3 = ransac_vm3.estimator_.coef_
    
    linskew_vm4 = lin_vm4.coef_
    tsenskew_vm4 = ts_vm4.coef_
    ransacskew_vm4 = ransac_vm4.estimator_.coef_    
    
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nThe data analysis for each regression model on VM1 is as follows:')
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nLinear Regression Model')
    model_analysis(x_vals[0], x_vm1, y_vm1, lin_vm1, linreg_vm1, linskew_vm1)
    f1.write('\nTheil-Sen Regression Model')
    model_analysis(x_vals[0], x_vm1, y_vm1, ts_vm1, tsreg_vm1, tsenskew_vm1)
    f1.write('\nRANSAC Regression Model')
    model_analysis(x_vals[0], x_vm1, y_vm1, ransac_vm1, ranreg_vm1, ransacskew_vm1)
    
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nThe data analysis for each regression model on VM2 is as follows:')
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nLinear Regression Model')
    model_analysis(x_vals[1], x_vm2, y_vm2, lin_vm2, linreg_vm2, linskew_vm2)
    f1.write('\nTheil-Sen Regression Model')
    model_analysis(x_vals[1], x_vm2, y_vm2, ts_vm2, tsreg_vm2, tsenskew_vm2)
    f1.write('\nRANSAC Regression Model')
    model_analysis(x_vals[1], x_vm2, y_vm2, ransac_vm2, ranreg_vm2, ransacskew_vm2) 
    
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nThe data analysis for each regression model on VM3 is as follows:')
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nLinear Regression Model')
    model_analysis(x_vals[2], x_vm3, y_vm3, lin_vm3, linreg_vm3, linskew_vm3)
    f1.write('\nTheil-Sen Regression Model')
    model_analysis(x_vals[2], x_vm3, y_vm3, ts_vm3, tsreg_vm3, tsenskew_vm3)
    f1.write('\nRANSAC Regression Model')
    model_analysis(x_vals[2], x_vm3, y_vm3, ransac_vm3, ranreg_vm3, ransacskew_vm3)
    
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nThe data analysis for each regression model on VM4 is as follows:')
    f1.write('\n-----------------------------------------------------------------')
    f1.write('\nLinear Regression Model')
    model_analysis(x_vals[3], x_vm4, y_vm4, lin_vm4, linreg_vm4, linskew_vm4)
    f1.write('\nTheil-Sen Regression Model')
    model_analysis(x_vals[3], x_vm4, y_vm4, ts_vm4, tsreg_vm4, tsenskew_vm4)
    f1.write('\nRANSAC Regression Model')
    model_analysis(x_vals[3], x_vm4, y_vm4, ransac_vm4, ranreg_vm4, ransacskew_vm4)    
    
    fig, ((vm1, vm2), (vm3, vm4)) = plt.subplots(2, 2)
    #fig, ((vm1, vm2), (vm3, vm4)) = plt.subplots(2, 2, sharex = 'col', sharey = 'row')
    #fig.title('VM Skew Analysis')
    #fig.xlabel('Host System Time Lapse in seconds (t_i - t_0)')
    #fig.ylabel('Observed VM Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    
    #plt.subplot(2, 1, 1)
    #vm1.plot(x_vm1, y_vm1, '.k')
    vm1.plot(vm1_xinliers, vm1_yinliers, '.k')
    vm1.plot(vm1_xoutliers, vm1_youtliers, '.r')    
    vm1.plot(x_vm1, linreg_vm1, '-m', label = 'Lin Reg Skew = %s' % linskew_vm1)
    vm1.plot(x_vm1, tsreg_vm1, '-b', label = 'TS Reg Skew = %s' % tsenskew_vm1)
    vm1.plot(x_vm1, ranreg_vm1, '-g', label = 'RANSAC Reg Skew = %s' % ransacskew_vm1)
    #vm1.plot([x0_vm1, x1_vm1], [y0_vm1, y1_vm1], c='r')
    #vm1.plot([tsx0_vm1, tsx1_vm1], [tsy0_vm1, tsy1_vm1], c='g')
    vm1.set_title('VM1-VMWare')
    vm1.set_ylabel('Observed VM Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    #vm1.axis([-5, 15, -0.005, 0.005])
    vm1.axis([0, 650, -0.03, 0.03])
    vm1.legend(loc = 'lower right')
    vm1.text(20, 0.02, r'# Inliers = %s' % len(vm1_xinliers))
            
    #plt.subplot(2, 1, 2)
    #vm2.plot(inliers_vm1, '.k')
    vm2.plot(vm2_xinliers, vm2_yinliers, '.k')
    vm2.plot(vm2_xoutliers, vm2_youtliers, '.r')    
    #vm2.plot(vm2_xinliers, vm2_yinliers, '-k', label = 'RANSAC Inliers')
    #vm2.plot(vm2_xoutliers, vm2_youtliers, '.r', label = 'RANSAC Outliers')
    vm2.plot(x_vm2, linreg_vm2, '-m', label = 'Lin Reg Skew = %s' % linskew_vm2)
    vm2.plot(x_vm2, tsreg_vm2, '-b', label = 'TS Reg Skew = %s' % tsenskew_vm2)
    vm2.plot(x_vm2, ranreg_vm2, '-g', label = 'RANSAC Reg Skew = %s' % ransacskew_vm2)    
    vm2.set_title('VM2-VMWare')
    #vm2.set_xlabel('Host System Time Lapse in seconds (t_i - t_0)')
    #vm2.set_ylabel('Observed VM Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    #vm2.axis([-5, 15, -0.005, 0.005])
    vm2.axis([0, 650, -0.03, 0.03])
    vm2.legend(loc = 'lower right')
    
    #plt.subplot(2, 1, 1)
    #vm3.plot(x_vm3, y_vm3, '.k')
    vm3.plot(vm3_xinliers, vm3_yinliers, '.k')
    vm3.plot(vm3_xoutliers, vm3_youtliers, '.r')    
    vm3.plot(x_vm3, linreg_vm3, '-m', label = 'Lin Reg Skew = %s' % linskew_vm3)
    vm3.plot(x_vm3, tsreg_vm3, '-b', label = 'TS Reg Skew = %s' % tsenskew_vm3)
    vm3.plot(x_vm3, ranreg_vm3, '-g', label = 'RANSAC Reg Skew = %s' % ransacskew_vm3)
    #vm3.plot([x0_vm3, x1_vm3], [y0_vm3, y1_vm3], c='r')
    #vm3.plot([tsx0_vm3, tsx1_vm3], [tsy0_vm3, tsy1_vm3], c='g')
    vm3.set_title('VM3-VirtualBox')
    vm3.set_xlabel('Host System Time Lapse in seconds (t_i - t_0)')
    vm3.set_ylabel('Observed VM3 Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    #vm3.axis([-5, 15, -0.005, 0.005])
    vm3.axis([0, 650, -0.03, 0.03])
    vm3.legend(loc = 'lower right')
            
    #plt.subplot(2, 1, 2)
    #vm4.plot(inliers_vm4, '.k')
    vm4.plot(vm4_xinliers, vm4_yinliers, '.k')
    vm4.plot(vm4_xoutliers, vm4_youtliers, '.r')    
    #vm4.plot(vm4_xinliers, vm4_yinliers, '-k', label = 'RANSAC Inliers')
    #vm4.plot(vm4_xoutliers, vm4_youtliers, '.r', label = 'RANSAC Outliers')
    vm4.plot(x_vm4, linreg_vm4, '-m', label = 'Lin Reg Skew = %s' % linskew_vm4)
    vm4.plot(x_vm4, tsreg_vm4, '-b', label = 'TS Reg Skew = %s' % tsenskew_vm4)
    vm4.plot(x_vm4, ranreg_vm4, '-g', label = 'RANSAC Reg Skew = %s' % ransacskew_vm4)    
    vm4.set_title('VM4-VirtualBox')
    vm4.set_xlabel('Host System Time Lapse in seconds (t_i - t_0)')
    #vm4.ylabel('Observed VM4 Offset in seconds\n(((T_i - T_0)/F) - (t_i - t_0))')
    #vm4.axis([-5, 15, -0.005, 0.005])
    vm4.axis([0, 650, -0.03, 0.03])
    vm4.legend(loc = 'lower right')    
        
    plt.show()
    
    return


#*************************
# ***** MAIN PROGRAM *****
#*************************

fn = 'Multi_Hypervisor_Trial_1.txt'
fn1 = 'Skew_Results_1.txt'

f = open(fn, 'r')
f1 = open(fn1, 'w')    

vm_timestamps = []
host_timestamps = []
x_values = []
y_values = []
    
vm_timestamps, host_timestamps = time_collect()

f1.write('***********\n')
f1.write('Test Run #1\n')
f1.write('***********\n')
x_values, y_values = skew_analysis(vm_timestamps, host_timestamps)

plot_data(x_values, y_values)

f.close()
f1.close()

#for i in range(2):
    #fn = 'Retest' + str(i+1) + '.txt'
    #fn1 = 'Skew_Results' + str(i+1) + '.txt'
    
    #f = open(fn, 'r')
    #f1 = open(fn1, 'w')    
    
    #vm_timestamps = []
    #host_timestamps = []
    #x_values = []
    #y_values = []
        
    #vm_timestamps, host_timestamps = time_collect()
    
    #f1.write('\n***********\n')
    #f1.write('Test Run #' + str(i+1) +'\n')
    #f1.write('***********\n')
    #x_values, y_values = skew_analysis(vm_timestamps, host_timestamps)

    #plot_data(x_values, y_values)
    
    #f.close()
    #f1.close()    
