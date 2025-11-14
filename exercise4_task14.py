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

# %% Dependencies
import os
import numpy as np
import pandas as pd
import load_profiles as lp           # <- provided with the exercise package
import pandapower_read_csv as ppcsv  # <- provided with the exercise package


path_data_set = '/Users/sannespakmo/Library/CloudStorage/OneDrive-Personal/Skole/9. semester/Fordypningsemne/Flexibility/Exercises/7703070'

# Filer i datasettet
filename_load_data_fullpath    = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')
filename_standard_overhead     = os.path.join(path_data_set,'standard_overhead_line_types.csv')
filename_reldata               = os.path.join(path_data_set,'reldata_for_component_types.csv')
filename_load_point            = os.path.join(path_data_set,'CINELDI_MV_reference_system_load_point.csv')

# Choices for the area and scaling
bus_i_subset    = [90, 91, 92, 96]
scaling_factor  = 10          # same as in Exercise 4
annual_growth   = 0.03
year_index      = 5           # year 6 => y = 5
scale_y6        = (1 + annual_growth)**year_index

# Grid limit in MW (used later in the model)
P_lim_MW = 4.0

# ================== READ STATIC TABLES (optional, used in later tasks) ==================
data_standard_overhead_lines = pd.read_csv(filename_standard_overhead, delimiter=';').set_index('type')
data_comp_rel                = pd.read_csv(filename_reldata, delimiter=';').set_index('main_type')
data_load_point              = pd.read_csv(filename_load_point, delimiter=';').set_index('bus_i')

# ================== READ NETWORK (pandapower) ==================
net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

# ================== READ AND MAP LOAD PROFILES ==================
# 28 February = day no. 59 of the year -> 31 (Jan) + 28 (Feb) = 59 => the helper function handles the zero-indexed hourly index
load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Use only 28 Feb as the representative day
repr_days = [31 + 28]  # [59]
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath, repr_days)
# profiles_mapped: rows = hours of the day (24), columns = buses (1-indexed)

# Scale by the nominal MW for each load in the grid data + increase with scaling_factor (Exercise 4)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])    # MW
load_time_series_subset = load_time_series_mapped[bus_i_subset] * scaling_factor

# Aggregated time series (MW) for the area
load_time_series_subset_aggr = load_time_series_subset.sum(axis=1)  # length 24

scale_y6 = (1.03**5)

# ================== BUILD INPUT DATA FOR OPTIMIZATION ==================
# Base_load in MW for year 6 (y=5)
Base_load_MW = load_time_series_subset_aggr.to_numpy() * scale_y6   # (24,)

# PV production set to zero (MW)
PV_prod_MW = np.zeros_like(Base_load_MW)

# Timer 0..23
Hours = np.arange(Base_load_MW.shape[0])

# ===== PRICE =====
# Try to read a price file if available; otherwise use a flat price (1000 NOK/MWh) to get started
candidate_price_files = [
    os.path.join(path_data_set, 'price_Feb28.csv'),
    os.path.join(path_data_set, 'prices.csv'),
]
Price_NOK_per_MWh = None
for pf in candidate_price_files:
    if os.path.isfile(pf):
        dfp = pd.read_csv(pf)
        # Make a cautious attempt to retrieve columns named "Price" or similar
        cand_cols = [c for c in dfp.columns if 'price' in c.lower()]
        if len(cand_cols) > 0:
            Price_NOK_per_MWh = dfp[cand_cols[0]].to_numpy()[:Base_load_MW.shape[0]]
            break

if Price_NOK_per_MWh is None:
    # fallback: flat price
    Price_NOK_per_MWh = np.full_like(Base_load_MW, 1000.0, dtype=float)

# Create dictionaries (Pyomo-friendly)
dict_Base_load = dict(zip(Hours, Base_load_MW))
dict_PV_prod   = dict(zip(Hours, PV_prod_MW))
dict_Prices    = dict(zip(Hours, Price_NOK_per_MWh))

# ================== QUICK CHECK ==================
print("=== Input for new model ===")
print(f"Hours: {Hours.shape} -> {Hours[:5]} ...")
print(f"Base_load_MW: shape {Base_load_MW.shape}, peak={Base_load_MW.max():.3f} MW, mean={Base_load_MW.mean():.3f} MW")
print(f"PV_prod_MW:   all zeros? {np.allclose(PV_prod_MW, 0.0)}")
print(f"Price (NOK/MWh): shape {Price_NOK_per_MWh.shape}, first={Price_NOK_per_MWh[0]:.1f}")
print(f"P_lim_MW = {P_lim_MW:.3f}")
print("Files read successfully. You can now plug this into the Pyomo model (MW/MWh units).")


#%% Read battery specifications
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

#Parse battery specification
capacity = 2 #parameters['Energy_capacity']
charging_power_limit = 1 #parameters["Power_capacity"]
discharging_power_limit = 1 #parameters["Power_capacity"]
charging_efficiency = parameters["Charging_efficiency"]
discharging_efficiency = parameters["Discharging_efficiency"]
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')

# Convert the various timeseries/profiles to numpy arrays
Hours = Hours = np.arange(Base_load_MW.shape[0])
Base_load = Base_load_MW # should be replaced with the load for 28 February
PV_prod = np.zeros_like(Base_load) # testData['PV_prod'].values
Price = testData['Price'].values

# Make dictionaries (for simpler use in Pyomo)
dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))

p_limit = 4
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

#Objective function
def OBJ(model):
    #return sum(model.Price[t]*(model.P_from_grid[t]-model.P_to_grid[t]) for t in model.T)
    return sum(model.Price[t]*(model.P_to_grid[t]-model.P_from_grid[t]) for t in model.T)
model.OBJ = pyo.Objective(rule=OBJ, sense=pyo.maximize)

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

# --- PLOT: Load with/without flexibility + SOC (dual axes, MW/MWh) ---
def plot_flex(hours, load_wo, load_w, soc, p_limit=4.0, title="Load with and without flexibility"):
    import numpy as np
    import matplotlib.pyplot as plt

    hours = np.asarray(hours)
    load_wo = np.asarray(load_wo, dtype=float)
    load_w  = np.asarray(load_w, dtype=float)
    soc     = np.asarray(soc, dtype=float)

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Left axis: loads (MW)
    ax1.plot(hours, load_wo, label='Load Without Flex')
    ax1.plot(hours, load_w,  label='Load With Flex')
    ax1.axhline(p_limit, linestyle=':', label=f'Congestion threshold ({p_limit:.0f} MW)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('MW')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Right axis: SOC (MWh)
    ax2 = ax1.twinx()
    ax2.plot(hours, soc, label='SOC (right)', color='tab:red', linestyle='--')
    ax2.set_ylabel('MWh')

    # Combined legend
    L1, lab1 = ax1.get_legend_handles_labels()
    L2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(L1 + L2, lab1 + lab2, loc='upper left')

    plt.tight_layout()
    plt.show()

# Hours for x-axis
hrs = df['hour'].values

# Load without flexibility (area base load in MW)
load_without_flex = df['base_load'].values

# Load with flexibility = net import the upstream grid sees (MW)
# You already computed this:
load_with_flex = df['net_load_grid'].values

# State of charge (MWh)
soc = df['E'].values

# Plot
plot_flex(hrs, load_without_flex, load_with_flex, soc, p_limit=4.0,
          title="Load with and without flexibility (BESS 1 MW / 2 MWh)")
