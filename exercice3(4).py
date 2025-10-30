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

# --- NET LOAD: compute, verify, and plot ------------------------------------
# Net load seen by the grid (positive = importing from grid, negative = exporting to grid)
df['net_load_grid'] = df['P_from_grid'] - df['P_to_grid']

# Alternative expression via load, PV and battery (should be identical)
df['net_load_alt'] = df['base_load'] - df['pv'] - df['P_d'] + df['P_c']

# Plot net load over the day
plt.figure(figsize=(10,4.2))
plt.plot(df['hour'], df['net_load_grid'], marker='o', label='Net load (P_from_grid - P_to_grid)')
plt.axhline(0, linewidth=1)
plt.xlabel('Time (h)')
plt.ylabel('Power (kW)')
plt.title('Net load profile (positive: import, negative: export)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import os
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Key figures -------------------------------------------------------------
peak_import = df['net_load_grid'].max()
t_peak_import = df.loc[df['net_load_grid'].idxmax(), 'hour']
peak_export = df['net_load_grid'].min()  # most negative
t_peak_export = df.loc[df['net_load_grid'].idxmin(), 'hour']
energy_import = df['net_load_grid'].clip(lower=0).sum()  # kWh over the day (1h resolution)
energy_export = (-df['net_load_grid']).clip(lower=0).sum()

print(f"Peak import: {peak_import:.2f} kW at hour {int(t_peak_import)}")
print(f"Peak export: {peak_export:.2f} kW at hour {int(t_peak_export)}")
print(f"Energy imported (24h): {energy_import:.2f} kWh")
print(f"Energy exported (24h): {energy_export:.2f} kWh")

path_data_set         = '/Users/sannespakmo/Library/CloudStorage/OneDrive-Personal/Skole/9. semester/Fordypningsemne/Flexibility/Exercises/7703070'
#path_data_set         = 'C:\\Users\\graff\\OneDrive\\Dokumenter\\CINELDI_MV_reference_system_v_2023-03-06' 

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')
filename_standard_overhead_lines = os.path.join(path_data_set,'standard_overhead_line_types.csv')
filename_reldata = os.path.join(path_data_set,'reldata_for_component_types.csv')
filename_load_point = os.path.join(path_data_set,'CINELDI_MV_reference_system_load_point.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 4

# Factor to scale the loads for this exercise compared with the base version of the CINELDI reference system data set
scaling_factor = 10

# Read standard data for overhead lines
data_standard_overhead_lines = pd.read_csv(filename_standard_overhead_lines, delimiter=';')
data_standard_overhead_lines.set_index(keys = 'type', drop = True, inplace = True)

# Read standard component reliability data
data_comp_rel = pd.read_csv(filename_reldata, delimiter=';')
data_comp_rel.set_index(keys = 'main_type', drop = True, inplace = True)

# Read load point data (incl. specific rates of costs of energy not supplied) for data
data_load_point = pd.read_csv(filename_load_point, delimiter=';')
data_load_point.set_index(keys = 'bus_i', drop = True, inplace = True)


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)


# %% Set up hourly normalized load time series for a representative day (task 2; this code is provided to the students)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Consider only the day with the peak load in the area (28 February)
repr_days = [31+28]

# Get relative load profiles for representative days mapped to buses of the CINELDI test network;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])

# ==== Exercise 4 – Battery operation with import cap (Alt. B) ====
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ------------------------------------------------------------------
# 1) DATA
# ------------------------------------------------------------------
# -- Prices (reuse your Exercise 3 price vector)
price_df = pd.read_csv('./profile_input.csv')   # has columns: Hours, Price (and PV/Base from Ex.3)
Hours = price_df['Hours'].values
Price = price_df['Price'].values

# -- Aggregated load for 28 February (MW, 24 points)
# Provide this Series from your earlier code; here we assume you have it in memory:
#   load_time_series_subset_aggr: pandas Series (24 values) in MW for 28 Feb
# If you don't have it here, you can also save it to CSV in the other script and read it back.
load_time_series_subset = load_time_series_mapped[bus_i_subset] * scaling_factor
load_time_series_subset_aggr = load_time_series_subset.sum(axis=1)

P_max = load_time_series_subset_aggr.max()

# Scale load to year 6 (starting at y=5) with 3% growth
growth = 0.03
scale_to_year6 = (1 + growth) ** 5
load_mw_y6 = load_time_series_subset_aggr.values * scale_to_year6           # MW
Base_load = (load_mw_y6 * 1000.0)                                           # kW

# -- PV is zero
PV_prod = np.zeros_like(Base_load)

# ------------------------------------------------------------------
# 2) BATTERY + LIMITS
# ------------------------------------------------------------------
# Battery: 1 MW / 2 MWh
P_batt_kw = 1000.0
E_batt_kwh = 2000.0
eta_c = 1.0       # set efficiencies = 1.0 unless you want specific values
eta_d = 1.0

# Net import limit: 4 MW -> 4000 kW
P_lim_kw = 4000.0

# ------------------------------------------------------------------
# 3) MODEL
# ------------------------------------------------------------------
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=range(len(Hours)), ordered=True)

# Params
model.Price   = pyo.Param(model.T, initialize=dict(enumerate(Price)), within=pyo.Reals)
model.Base    = pyo.Param(model.T, initialize=dict(enumerate(Base_load)), within=pyo.Reals)
model.PV      = pyo.Param(model.T, initialize=dict(enumerate(PV_prod)), within=pyo.Reals)
model.cap     = pyo.Param(initialize=E_batt_kwh)      # kWh
model.eta_c   = pyo.Param(initialize=eta_c)
model.eta_d   = pyo.Param(initialize=eta_d)
model.Pc_max  = pyo.Param(initialize=P_batt_kw)       # kW
model.Pd_max  = pyo.Param(initialize=P_batt_kw)       # kW
model.P_lim   = pyo.Param(initialize=P_lim_kw)        # kW

# Vars
model.P_c = pyo.Var(model.T, bounds=(0, model.Pc_max))            # kW
model.P_d = pyo.Var(model.T, bounds=(0, model.Pd_max))            # kW
model.P_from = pyo.Var(model.T, within=pyo.NonNegativeReals)      # kW
model.P_to   = pyo.Var(model.T, within=pyo.NonNegativeReals)      # kW
model.E      = pyo.Var(model.T, bounds=(0, model.cap))            # kWh

# Objective: minimize net energy cost (== maximize arbitrage revenue)
def OBJ(m):
    return sum(m.Price[t] * (m.P_from[t] - m.P_to[t]) for t in m.T)
model.OBJ = pyo.Objective(rule=OBJ, sense=pyo.minimize)

# Constraints
def power_balance(m, t):
    return m.P_from[t] + m.P_d[t] + m.PV[t] == m.Base[t] + m.P_c[t] + m.P_to[t]
model.power_balance = pyo.Constraint(model.T, rule=power_balance)

def energy_balance(m, t):
    if t == m.T.first():
        return m.E[t] == 0.0
    return m.E[t] == m.E[t-1] + m.eta_c*m.P_c[t-1] - (1/m.eta_d)*m.P_d[t-1]
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance)

def end_energy(m):
    return m.E[m.T.last()] == 0.0
model.end_energy = pyo.Constraint(rule=end_energy)

# Ensure we don't discharge more than available energy in 1h:
def max_discharge_energy(m, t):
    return m.P_d[t] <= m.E[t] * m.eta_d
model.max_discharge_energy = pyo.Constraint(model.T, rule=max_discharge_energy)

# Net import cap each hour
def import_cap(m, t):
    return m.P_from[t] <= m.P_lim
model.import_cap = pyo.Constraint(model.T, rule=import_cap)

# (Optional) prevent simultaneous buy & sell (bilinear; requires Gurobi with NonConvex=2)
def no_buy_and_sell(m, t):
    return m.P_to[t] * m.P_from[t] == 0
model.no_buy_and_sell = pyo.Constraint(model.T, rule=no_buy_and_sell)

# ------------------------------------------------------------------
# 4) SOLVE
# ------------------------------------------------------------------
opt = SolverFactory('gurobi')
results = opt.solve(model, tee=False)

# ------------------------------------------------------------------
# 5) RESULTS
# ------------------------------------------------------------------
df = pd.DataFrame({
    'hour'        : Hours,
    'price'       : [pyo.value(model.Price[t]) for t in model.T],
    'base_load'   : [pyo.value(model.Base[t])  for t in model.T],
    'pv'          : [pyo.value(model.PV[t])    for t in model.T],
    'P_c'         : [pyo.value(model.P_c[t])   for t in model.T],
    'P_d'         : [pyo.value(model.P_d[t])   for t in model.T],
    'P_from_grid' : [pyo.value(model.P_from[t]) for t in model.T],
    'P_to_grid'   : [pyo.value(model.P_to[t])   for t in model.T],
    'E'           : [pyo.value(model.E[t])     for t in model.T],
})
df['net_import'] = df['P_from_grid'] - df['P_to_grid']   # kW

total_cost = (df['price'] * (df['P_from_grid'] - df['P_to_grid'])).sum()
print(f"Total arbitrage cost (negative = net revenue): {total_cost:.2f}")

violations = (df['P_from_grid'] > P_lim_kw).sum()
print(f"Hours violating P_lim=4 MW: {violations} (should be 0 if battery can enforce the cap)")

# ------------------------------------------------------------------
# 6) PLOTS
# ------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(df['hour'], df['base_load']/1000, label='Aggregated load (MW)')
plt.plot(df['hour'], df['net_import']/1000, label='Net import (MW)')
plt.axhline(P_lim_kw/1000, ls='--', label='P_lim = 4 MW')
plt.xlabel('Hour')
plt.ylabel('MW')
plt.title('Net import vs. 4 MW limit')
plt.grid(True, alpha=0.3)
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(df['hour'], df['P_c'], label='Charge (kW)')
plt.plot(df['hour'], df['P_d'], label='Discharge (kW)')
plt.xlabel('Hour'); plt.ylabel('kW'); plt.title('Charge/Discharge schedule')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(df['hour'], df['E'], label='SoC (kWh)')
plt.xlabel('Hour'); plt.ylabel('kWh'); plt.title('Battery SoC (2 MWh)')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
