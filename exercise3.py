# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:30:27 2023

@author: merkebud, ivespe

Intro script for Exercise 3 ("Scheduling flexibility resources") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as pyo
import time

#%% Read battery specifications
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

#Parse battery specification
capacity=parameters['Energy_capacity']
charging_power_limit=parameters["Power_capacity"]
discharging_power_limit=parameters["Power_capacity"]
charging_efficiency=parameters["Charging_efficiency"]
discharging_efficiency=parameters["Discharging_efficiency"]
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')

# Convert the various timeseries/profiles to numpy arrays
Hours = testData['Hours'].values
Base_load = testData['Base_load'].values
PV_prod = testData['PV_prod'].values
Price = testData['Price'].values

# Make dictionaries (for simpler use in Pyomo)
dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))

p_limit = 6.05
# %%

model = pyo.ConcreteModel()

model.T = pyo.Set(initialize=testData.index, ordered=True)


#Parameters
model.Base_load = pyo.Param(model.T, initialize=Base_load)
model.PV_prod = pyo.Param(model.T, initialize=PV_prod)
model.Price = pyo.Param(model.T, initialize=Price)

model.cap = pyo.Param(initialize=capacity)
model.eta_c = pyo.Param(initialize=charging_efficiency)
model.eta_d = pyo.Param(initialize=discharging_efficiency)
model.P_c_max = pyo.Param(initialize=charging_power_limit)
model.P_d_max = pyo.Param(initialize=discharging_power_limit)
model.p_lim = pyo.Param(initialize = p_limit)   

#Variables
model.P_c =pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0,model.P_c_max)) #Charging power
model.P_d =pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0,model.P_d_max)) #Discharging power
model.P_from_grid = pyo.Var(model.T, within=pyo.NonNegativeReals) #Power bought from grid
model.P_to_grid = pyo.Var(model.T, within=pyo.NonNegativeReals) #Power sold to grid
model.E = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0,model.cap)) #Energy stored in battery 

#Objective function: Minimize cost of electricity
def OBJ(model):
    return sum(model.Price[t]*(model.P_from_grid[t]-model.P_to_grid[t]) for t in model.T)
model.OBJ = pyo.Objective(rule=OBJ, sense=pyo.minimize)

#Constraints

def power_balance_rule(model, t):
    return model.P_from_grid[t] + model.P_d[t] + model.PV_prod[t] == model.Base_load[t] + model.P_c[t] + model.P_to_grid[t]
model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

def energy_balance_rule(model, t): 
    if t == model.T.first():
        return model.E[t] == 0 #battery state of charge zero at the beginning and the end of the day
    else:
        return model.E[t] == model.E[t-1] + model.eta_c*model.P_c[t-1] - (1/model.eta_d)*model.P_d[t-1] #t-1 here??
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

def end_energy_rule(model):
    return model.E[model.T.last()] == 0
model.end_energy = pyo.Constraint(rule=end_energy_rule)

def max_discharge_rule(model, t):
    return model.P_d[t] <= model.E[t]*model.eta_d #Multiply with efficiency?
model.max_discharge = pyo.Constraint(model.T, rule=max_discharge_rule) 

def sell_buy_rule(model, t):
    return model.P_to_grid[t]* model.P_from_grid[t] ==0
model.sell_buy = pyo.Constraint(model.T, rule=sell_buy_rule) #make sure we don't buy and sell at the same time

def charge_limit_rule(model, t):
    return model.P_c[t] <= charging_power_limit
model.charge_limit = pyo.Constraint(model.T, rule=charge_limit_rule)

def discharge_limit_rule(model, t):
    return model.P_d[t] <= discharging_power_limit
model.discharge_limit = pyo.Constraint(model.T, rule=discharge_limit_rule)

def battery_capacity_rule(model, t):
    return model.E[t] <= capacity
model.battery_capacity = pyo.Constraint(model.T, rule=battery_capacity_rule)

def to_grid_limit_rule(model, t):
    return 0 <= model.P_to_grid[t]
model.to_grid_limit = pyo.Constraint(model.T, rule=to_grid_limit_rule)


def from_grid_limit_rule(model, t):
    return model.P_from_grid[t] <= model.p_lim
model.from_grid_limit = pyo.Constraint(model.T, rule=from_grid_limit_rule)

#%% Solve the optimization problem
opt = SolverFactory('gurobi') #you can also use 'glpk'

results = opt.solve(model, tee=True) #set tee=True if you want to see the solver output

# --- FETCH RESULTS ---------------------------------------------------
df = pd.DataFrame({
    'hour'        : Hours,  # 0..23
    'price'       : [pyo.value(model.Price[t]) for t in model.T],
    'base_load'   : [pyo.value(model.Base_load[t]) for t in model.T],
    'pv'          : [pyo.value(model.PV_prod[t]) for t in model.T],
    'P_c'         : [pyo.value(model.P_c[t]) for t in model.T],
    'P_d'         : [pyo.value(model.P_d[t]) for t in model.T],
    'P_from_grid' : [pyo.value(model.P_from_grid[t]) for t in model.T],
    'P_to_grid'   : [pyo.value(model.P_to_grid[t]) for t in model.T],
    'E'           : [pyo.value(model.E[t]) for t in model.T],
    
})
total_cost = (df['price'] * (df['P_from_grid'] - df['P_to_grid'])).sum()

print(f"Total cost for the day: {total_cost:.2f} (same currency/unit as 'Price')")

# --- PLOT: charge/discharge schedule + SoC ----------------------------------
plt.figure(figsize=(10,4.2))
plt.plot(df['hour'], df['P_c'], label='Charging P_c (kW)')
plt.plot(df['hour'], df['P_d'], label='Discharging P_d (kW)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.title('Battery charge/discharge schedule (24h)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,4.2))
plt.plot(df['hour'], df['P_from_grid'], label='P_from_grid (kW)')
plt.plot(df['hour'], df['P_to_grid'], label='P_to_grid (kW)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.title('Load profile (24h)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4.2))
plt.plot(df['hour'], df['E'],  label='SoC E (kWh)')
plt.xlabel('Time')
plt.ylabel('Energy (kWh)')
plt.title('Battery energy level (SoC) throughout the day')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- PLOT: charge/discharge schedule + SoC (dual axes) -----------------------------
fig, ax1 = plt.subplots(figsize=(10,4.2))

# Left axis: power (kW)
line_pc, = ax1.plot(df['hour'], df['P_c'], marker='o', linestyle='-', label='Charging P_c (kW)')
line_pd, = ax1.plot(df['hour'], df['P_d'], marker='o', linestyle='-', label='Discharging P_d (kW)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (kW)')
ax1.set_title('Battery charge/discharge schedule (24h) + SoC')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df['hour'])

# Right axis: SoC (kWh)
ax2 = ax1.twinx()
line_soc, = ax2.plot(df['hour'], df['E'], marker='*', linestyle='--', label='SoC (kWh)', color='tab:green')
ax2.set_ylabel('Energy (kWh)')
ax2.set_ylim(0, max(1.0, df['E'].max()*1.05))  # small margin at the top

# Shared legend
lines = [line_pc, line_pd, line_soc]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

fig.tight_layout()
plt.show()

  
