# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:30:12 2022

@author: yangt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pyfmi import load_fmu
from datetime import datetime
from pytz import timezone

# time tackle
from datetime import datetime
import time
from pytz import timezone


#%% define functions
# get measurement
def get_measurement(fmu,names,t, dt):

    dic = {}
    for name in names:
        dic[name] = fmu.get(name)
    # specify clock
    dic['time'] = t

    #submeter for HVAC power
    PTot_sum = fmu.get('EC_CC')+fmu.get('EC_HC')+fmu.get('EC_HC_De')+fmu.get('EC_HC_Cr')+\
        fmu.get('EC_Fan_sa')+fmu.get('EC_Fan_oas')+fmu.get('EC_Fan_oae')+fmu.get('EC_OA_Rec')+\
            fmu.get('EC_HC_Supp')+fmu.get('EC_Fan_HPW')     
    dic['PTot'] = PTot_sum*2.7777777777778E-7*(3600./dt)
    #1J = 2.7778e-7 kWh

    # return a pandas data frame
    return pd.DataFrame(dic,index=[dic['time']])



#%% simulation setup
fmu_name = "Houston_2021_ff.fmu"

start = 212*24*3600. #start time
ndays = 7
end = start + ndays*24*3600. #end time
nsteps_h = 1
dt = 3600./nsteps_h

# define outputs
# read measurements
measurement_names = ['T_id','EC_HVAC_ems','T_od']
measurement_mpc = pd.DataFrame()
measurement_base = pd.DataFrame()
measurement_boundary = pd.DataFrame()

## for sav file name
# specific name
specific = '7days-higher_peak_2-p2-a0b0.01_' #if any, format is 'a0.3-5_', do not forget"_"
plot_title= 'higher_peak_2-p2-a0b0.01'

price_tou = [0.077926, 0.077926, 0.077926, 0.077926, 
        0.077926, 0.077926, 0.077926, 0.077926, 
        0.077926, 0.077926, 0.077926, 0.077926, 
        0.077926, 0.38963, 0.38963, 0.38963, 
        0.38963, 0.38963, 0.38963, 0.38963, 
        0.077926, 0.077926, 0.077926, 0.077926]

#%% baseline simulation

## load virtual building model
baseline = load_fmu(fmu_name)

## fmu settings
opts = baseline.simulate_options()
opts['ncp'] = int((end-start)/dt)

## initialize
baseline.initialize(start_time = start, stop_time = end)

##  run baselin with given setpoints
ts = start

while ts < end:
  baseline.set("cooling_stp",24.4)
  baseline.do_step(current_t=ts, step_size=dt, new_step=True)
  
  # store measurement
  measurement_step = get_measurement(baseline,measurement_names,ts, dt)
  measurement_base = measurement_base.append(measurement_step)
  # update clock
  ts += dt

# free instance for a second call
baseline.free_instance()

#%% boundary simulation

## load virtual building model
boundary = load_fmu(fmu_name)

## fmu settings
opts = boundary.simulate_options()
opts['ncp'] = int((end-start)/dt)

## initialize
boundary.initialize(start_time = start, stop_time = end)

##  run baselin with given setpoints
ts = start

while ts < end:
  boundary.set("cooling_stp",22.2)
  boundary.do_step(current_t=ts, step_size=dt, new_step=True)
  
  # store measurement
  measurement_step = get_measurement(boundary,measurement_names,ts, dt)
  measurement_boundary = measurement_boundary.append(measurement_step)
  # update clock
  ts += dt

# free instance for a second call
boundary.free_instance()

#%% MPC final simulation

## read optimal control inputs
with open('u_opt.json') as f:
  opt = json.load(f)

t_opt = opt['t_opt']
u_opt = opt['u_opt']
P_pred = opt['P_pred_opt']
Tz_pred = opt['Tz_pred_opt']

## Load virtual building model
mpc = load_fmu(fmu_name)

## fmu settings
opts = mpc.simulate_options()
opts['ncp'] = int((end-start)/dt)

## construct optimal input for fmu
## initialize
mpc.initialize(start_time = start, stop_time = end)

##  run mpc case with given optimal setpoints
for i, ts in zip(np.arange(len(t_opt)),t_opt):
  mpc.set("cooling_stp",u_opt[i])
  mpc.do_step(current_t=ts, step_size=dt, new_step=True)
  
  # store measurement
  measurement_step = get_measurement(mpc,measurement_names,ts, dt)
  measurement_mpc = measurement_mpc.append(measurement_step)
  
mpc.free_instance()

#%% plot - data preparation
## x values
x_range = measurement_base['time'] # for timestep
x_range_day = np.array([i+1 for i in range(ndays)]) # for days

## price timestep
price_plot = np.array(price_tou*ndays)

## energy
# energy timestep
energy_mpc = np.array(measurement_mpc['PTot'])/(3600./dt)
energy_base =np.array(measurement_base['PTot'])/(3600./dt)
energy_boundary = np.array(measurement_boundary['PTot'])/(3600./dt)
# power saving timestep
power_savings_mpc = (measurement_base['PTot']-measurement_mpc['PTot'])/measurement_base['PTot']
power_savings_mpc = power_savings_mpc.fillna(0)
power_savings_boundary = (measurement_base['PTot']-measurement_boundary['PTot'])/measurement_base['PTot']
power_savings_boundary = power_savings_boundary.fillna(0)

## cost
# cost timestep
cost_mpc = price_plot*energy_mpc
cost_base = price_plot*energy_base
cost_boundary = price_plot*energy_boundary
# cost per day
ts_per_day = 24*nsteps_h
cost_mpc_days = [sum(cost_mpc[i*ts_per_day:(i+1)*ts_per_day]) for i in range (ndays)]
cost_base_days = [sum(cost_base[i*ts_per_day:(i+1)*ts_per_day]) for i in range (ndays)]
cost_boundary_days = [sum(cost_boundary[i*ts_per_day:(i+1)*ts_per_day]) for i in range (ndays)]

## cost saving
# cost saving timestep
cost_savings_mpc = (cost_base-cost_mpc)/cost_base
cost_savings_boundary = (cost_base-cost_boundary)/cost_base
# cost saving per day
cost_savings_mpc_days = (np.array(cost_base_days)-np.array(cost_mpc_days))/np.array(cost_base_days) *100
cost_savings_boundary_days = (np.array(cost_base_days)-np.array(cost_boundary_days))/np.array(cost_base_days) *100

#%% plot - configuration
## x ticks
# x ticks resolution
xresolution = 12
xresolution_2 = 24
# x ticks
xticks=np.arange(start,end+1,xresolution*3600)
xticks_2=np.arange(start,end+1,xresolution_2*3600)

## x lables
# x label location
ha = ['right', 'center', 'left']
# x labels 
xticks_time = pd.to_datetime(xticks, unit='s')
format_1 = "%m-%d %H:%M"
xticks_label_time = xticks_time.strftime(format_1)

xticks_time_2 = pd.to_datetime(xticks_2, unit='s')
format_2 = "%m-%d"
xticks_label_time_2 = xticks_time_2.strftime(format_2)

## line_width
lw_0 = 2 

## bar width
width = 0.2

## colors
c_blue = '#187498' 
c_orange = '#FFB562'
c_red = '#EB5353'
c_green = '#367E18'

#%% plot - plot

## canvas
fig,([ax1,ax1b],[ax2,ax5],[ax3,ax4])=plt.subplots(nrows=3, ncols=2,figsize=(16,10))

## title
plt.suptitle(plot_title,fontsize=20)

## subplot 1
# plot
ax1.plot(x_range,price_plot,label='price',c=c_green)
# x axis
ax1.set_xticks(xticks,xticks_label_time,rotation=30,ha=ha[0])
# y axis
ax1.set_ylabel('Price ($/kWh)')
ax1.set_ylim([0,0.6])
# legend
ax1.legend(loc="upper left",fontsize=12)
# grid
ax1.grid(True)

## subplot 1b
# plot
ax1b.plot(x_range,price_plot,label='price',c=c_green)
# x axis
ax1b.set_xticks(xticks,xticks_label_time,rotation=30,ha=ha[0])
# y axis
ax1b.set_ylabel('Price ($/kWh)')
ax1b.set_ylim([0,0.6])
# legend
ax1b.legend(loc="upper left",fontsize=12)
# grid
ax1b.grid(True)

## subplot 2
# plot
ax2.plot(x_range,cost_base,c_blue,ls='-',label='baseline',lw=lw_0)
ax2.plot(x_range,cost_boundary,c_orange,ls='-',label='boundary',lw=lw_0)
ax2.plot(x_range,cost_mpc,c_red,ls='-',label='adaptive control',lw=lw_0)
# x axis
ax2.set_xticks(xticks,xticks_label_time,rotation=30,ha=ha[0])
# y axis
ax2.set_ylabel('Cost ($)')
# legend
ax2.legend(loc="lower left",fontsize=12)
# grid
ax2.grid(True)

## subplot 3
# plot
rects1 = ax3.bar(x_range_day-width, cost_mpc_days, width, color=c_red,\
                 label='adaptive control')
rects2 = ax3.bar(x_range_day, cost_base_days, width,\
                 color=c_blue, label='baseline')
rects3 = ax3.bar(x_range_day+width, cost_boundary_days, width,\
                 color=c_orange, label='boundary')
# bar label
ax3.bar_label(rects1,padding=1,labels=[f'${x:,.1f}' for x in rects1.datavalues])
ax3.bar_label(rects2,padding=12,labels=[f'${x:,.1f}' for x in rects2.datavalues])
ax3.bar_label(rects3,padding=2,labels=[f'${x:,.1f}' for x in rects3.datavalues])
# x axis
ax3.set_xticks(x_range_day,list(xticks_label_time_2)[:-1])    
ax3.set_xlabel('Absolute Cost',fontsize=16)
# y axis
ax3.set_ylabel('Cost ($)')
# legend
ax3.legend(loc="lower left",fontsize=12)
# grid
ax3.grid(True)


## subplot 4
# plot
rects1 = ax4.bar(x_range_day-width/2, cost_savings_mpc_days, width, color=c_red,\
                 label='adaptive control')
rects3 = ax4.bar(x_range_day+width/2, cost_savings_boundary_days, width,\
                 color=c_orange, label='boundary')
# bar label
ax4.bar_label(rects1,padding=2,labels=[f'{x:,.1f}%' for x in rects1.datavalues])
ax4.bar_label(rects3,padding=2,labels=[f'{x:,.1f}%' for x in rects3.datavalues])
# x axis
ax4.set_xticks(x_range_day,list(xticks_label_time_2)[:-1])    
ax4.set_xlabel('Relative Cost',fontsize=16)
# y axis
ax4.set_ylabel('Cost saving (%)')
ax4.set_ylim([-35,10])
# legend
ax4.legend(loc="lower left",fontsize=12)
# grid
ax4.grid(True)

## subplot 5
# plot
ax5.plot(x_range,cost_savings_boundary,c_orange,ls='-',label='boundary',lw=lw_0)
ax5.plot(x_range,cost_savings_mpc,c_red,ls='-',label='adaptive control',lw=lw_0)
# x axis
ax5.set_xticks(xticks,xticks_label_time,rotation=30,ha=ha[0])
# y axis
ax5.set_ylabel('Cost saving (%)')
# legend
ax5.legend(loc="lower left",fontsize=12)
# grid
ax5.grid(True)

## tight layout
plt.tight_layout(pad=1.0)

#%% save figure
## time stamp
format_1 = "%Y-%m-%d %H:%M:%S %Z%z"
format_2 = "%m%d%H%M"
# US/Central time zone
now_utc = datetime.now(timezone('UTC'))
now_myzone = now_utc.astimezone(timezone('US/Central'))
time_stamp = now_myzone.strftime(format_2)
print(time_stamp)

## save
plt.savefig('mpc-vs-rbc_tou_'+specific+time_stamp+'.png',dpi=300)
plt.show()