# -*- coding: utf-8 -*-
"""
Created on 2023-10-10

@author: ivespe

Intro script for Exercise 4 ("Battery energy storage system in the grid vs. grid investments") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""


# %% Dependencies

import pandas as pd
import os
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import numpy as np

# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)

#path_data_set         = '/Users/sannespakmo/Library/CloudStorage/OneDrive-Personal/Skole/9. semester/Fordypningsemne/Flexibility/Exercises/7703070'
path_data_set         = 'C:\\Users\\graff\\OneDrive\\Dokumenter\\CINELDI_MV_reference_system_v_2023-03-06' 

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


# %% Aggregate the load demand in the area

# Aggregated load time series for the subset of load buses
load_time_series_subset = load_time_series_mapped[bus_i_subset] * scaling_factor
load_time_series_subset_aggr = load_time_series_subset.sum(axis=1)

P_max = load_time_series_subset_aggr.max()

#TASK 2

growth = 0.03     # 3% annual growth
P_limit = 4.0     # MW
years = 10        # or set any horizon you want

# Peak each year (geometric growth from today's representative peak)
peak_by_year = pd.Series(
    [P_max * ((1 + growth) ** y) for y in range(years)],
    index=pd.Index(range(1, years + 1), name='Year')
)

# Find the first year where the peak exceeds the 4 MW limit
first_violation_year = peak_by_year[peak_by_year > P_limit].index.min()
# Plot
plt.figure()

x = np.r_[peak_by_year.index, peak_by_year.index[-1] + 1]
y = np.r_[peak_by_year.values, peak_by_year.values[-1]]

plt.step(x, y, where='post', label='Annual peak (MW)')
plt.axhline(P_limit, linestyle='--', label='Limit 4 MW')

plt.title('Task 2: Annual peak with 3% growth (step)')
plt.xlabel('Year')
plt.ylabel('MW')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#TASK 3

cost_per_km = 759408 #NOK/km for FeAl 70
length_km = 20.0

capex_NOK = cost_per_km * length_km
#print(f"Task 3: Total investment = {capex_NOK:,.0f} NOK")

#TASK 4
r = 0.04
investment_year = 2

years_to_discount = investment_year - 1
PV_capex = capex_NOK / ((1 + r) ** years_to_discount)

#print(f"Task 4: PV of investment (year {investment_year} start, 4%): {PV_capex:,.0f} NOK")

#TASK 5
y_end = 20
T_life = 40


y_end_cor = 20-1  # investment at start of year 2
remaining_fraction =  1 - y_end_cor / T_life

# Residual value and its PV
residual_value = capex_NOK * remaining_fraction
PV_residual = residual_value / ((1 + r) ** y_end)

# Corrected PV
PV_corrected = PV_capex - PV_residual

print(f"Residual value (undiscounted, end of year {y_end}): {residual_value:,.0f} NOK")
print(f"PV(residual): {PV_residual:,.0f} NOK")
print(f"Corrected PV (PV_inv - PV_res): {PV_corrected:,.0f} NOK")

#TASK 6
growth = 0.03
limit_with_bess = 5.0  # 4 + 1 MW

# smallest y with P_max*(1+g)^y > 5
y_cross = int(np.ceil(np.log(limit_with_bess / P_max) / np.log(1 + growth)))

growth = 0.03           # 3% annual growth
P_limit = 4.0           # MW (original)
BESS_P = 1.0            # MW
limit_eff = P_limit + BESS_P  # 5 MW with battery
years_show = 10         # show years 0..10

# Annual peaks from your current P_max (year 0 baseline)
peaks = [P_max * ((1 + growth) ** y) for y in range(years_show + 1)]

# First year that exceeds the effective limit
congestion_year = next((y for y, p in enumerate(peaks) if p > limit_eff), None)

# Build step arrays so the last step is visible
x = np.r_[np.arange(0, years_show + 1), years_show + 1]
y = np.r_[peaks, peaks[-1]]

plt.figure()
plt.step(x, y, where='post', label='Peak load demand')
plt.axhline(limit_eff, linestyle='--', color='red', label='Effective threshold (5 MW)')

if congestion_year is not None:
    plt.axvline(congestion_year, linestyle='--', color='green', label=f'Congestion year = {congestion_year}')

plt.title('Peak load development with a battery')
plt.xlabel('Year')
plt.ylabel('Peak load [MW]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

capex = 759_408 * 20.0     # 15,188,160 NOK
r = 0.04
y_end = 20
T_life = 40
investment_year = 1

c_inv_pv = capex_NOK / ((1 + r) ** 1)
print("Task 6: Present value of grid investment:", c_inv_pv)

y_end_cor = 20-1  # investment at start of year 9
remaining_fraction =  1 - y_end_cor / T_life

# Residual value and its PV
residual_value = capex_NOK * remaining_fraction
PV_residual = residual_value / ((1 + r) ** y_end)


print(f"PV(residual): {PV_residual:,.0f} NOK")
PV_corrected = c_inv_pv - PV_residual
print(f"Corrected PV (PV_inv - PV_res): {PV_corrected:,.0f} NOK")


#TASK 7

# Assumptions
cost_per_mwh = 2000.0     # NOK/MWh
days_per_year = 20        # number of similar days in the year
P_limit = 4.0             # MW (grid limit)
P_bess = 1.0              # MW (maximum hourly load shifting from the battery)
growth = 0.03             # annual growth
planning_horizon = y_end  # e.g., 20 years, reusing the variable from earlier


P_max_base = float(load_time_series_subset_aggr.max())      # MW on representative day (year 1)
reinforce_year = None
for y in range(1, planning_horizon + 1):
    peak_y = P_max_base * ((1 + growth) ** (y - 1))
    if peak_y > (P_limit + P_bess):   # exceeds 5 MW
        reinforce_year = y      # reinforce at the start of the next year
        break

annual_costs = {}
for y in range(1, planning_horizon + 1):
    if (reinforce_year is not None) and (y >= reinforce_year):
        annual_costs[y] = 0.0
        continue

    growth_factor = (1 + growth) ** (y - 1)
    load_y = load_time_series_subset_aggr * growth_factor  # MW per time

    excess = np.maximum(load_y - P_limit, 0.0)             # MW per time
    shifted_per_hour = np.minimum(excess, P_bess)          # MW per time (cap 1 MW)
    E_shift_day = float(shifted_per_hour.sum())            # MWh for the entire day

    annual_costs[y] = E_shift_day * cost_per_mwh * days_per_year

print("TASK 7: Yearly operating costs for congestion management (NOK/year)")
if reinforce_year is None:
    print("- No reinforcement in the horizon; services are purchased every year.")
else:
    print(f"- First year with peak > 5 MW: year {reinforce_year-1}.")
    print(f"- Reinforcement at the start of year {reinforce_year} (no purchases from this year onwards).")

for y in range(1, reinforce_year + 1):
    print(f"Year {y:2d}: {annual_costs[y]:,.0f} NOK")

#Task 8

length_km      = 20.0      # main feeder length
P_avg_year1    = 1.841     # MW (average load in year 1)
growth         = 0.03      # 3% annual growth
planning_horizon = 10   # e.g., 20

lambda_perm_per_km_year = 0.0397  # <-- set: faults per km per year (e.g., 0.05)
repair_time_hours       = 3  # <-- set: hours per fault (e.g., 8.0)

faults_per_year = lambda_perm_per_km_year * length_km

print("\nTASK 8: Expected Energy Not Supplied (EENS) per year (MWh)")
eens = []
for y in range(1, planning_horizon + 1):
    P_avg_y = P_avg_year1 * ((1 + growth) ** (y - 1))   # MW
    EENS_y  = P_avg_y * faults_per_year * repair_time_hours  # MWh/year
    eens.append(EENS_y)
    print(f"Year {y:2d}: EENS = {EENS_y:.2f} MWh")

print(data_load_point.columns)

# Task 9

# Inputs
P_avg_year1 = 1.841          # MW (area average in year 1)
growth = 0.03                # 3% annual growth
years = 9
length_km = 20.0
lambda_perm_100km = 3.97     # faults per 100 km-year (permanent)
r_hours = 3.0                # hours per permanent fault

# Faults per year on the 20 km main feeder
faults_per_year = lambda_perm_100km * (length_km / 100.0)   # = 0.794

cost_col = 'c_NOK_per_kWh_4h'

base_bus_mw = net.load.loc[bus_i_subset, 'p_mw'] * scaling_factor
bus_weights = base_bus_mw / base_bus_mw.sum()               # weights sum to 1
spec_cost_bus = data_load_point.loc[bus_i_subset, cost_col].astype(float)
avg_spec_cost = float((spec_cost_bus * bus_weights).sum())  # NOK/kWh

for y in range(0, years + 1):
    P_avg_y = P_avg_year1 * ((1 + growth) ** (y))       # MW
    ENS_y_MWh = P_avg_y * r_hours * faults_per_year         # MWh/year
    CENS_y_NOK = ENS_y_MWh * 1000.0 * avg_spec_cost         # NOK/year
    print(f"Year {y:2d}: CENS = {CENS_y_NOK:,.0f} NOK")

# Task 10    

P_avg0 = 1.841            # MW, average load in year 0
growth = 0.03             # 3% annual growth
years = 10                # prints Year 0..9
length_km = 20.0          # main feeder length
lambda_perm_100km = 3.97  # permanent faults per 100 km-year (Overhead line 1–22 kV)
r_hours = 3.0             # hours per fault
E_B = 2.0                 # MWh, battery energy per outage (usable)

faults_per_year = lambda_perm_100km * (length_km / 100.0)   # = 0.794


cost_col = 'c_NOK_per_kWh_4h'  


base_bus_mw = net.load.loc[bus_i_subset, 'p_mw'] * scaling_factor
weights = base_bus_mw / base_bus_mw.sum()
spec_cost_bus = data_load_point.loc[bus_i_subset, cost_col].astype(float)
avg_spec_cost = float((spec_cost_bus * weights).sum())  # NOK/kWh

delta_cens = faults_per_year * E_B * 1000.0 * avg_spec_cost

print(f"Using specific cost column: {cost_col}")
print(f"faults/year = {faults_per_year:.3f}, outage duration = {r_hours:.1f} h, battery = {E_B:.1f} MWh")
print(f"Constant annual reduction from battery: {delta_cens:,.0f} NOK\n")

for y in range(years):
    P_avg_y = P_avg0 * ((1 + growth) ** y)                 # MW
    eens_B_MWh = faults_per_year * max(P_avg_y * r_hours - E_B, 0.0)
    cens_B_NOK = eens_B_MWh * 1000.0 * avg_spec_cost
    print(f"Year {y:2d}: CENS(B) = {cens_B_NOK:,.0f} NOK")

#task 12 - Socio-economic cost for solution A

#socialeco_cost_A = PV_corrected + sum(CENS_y_NOK) må bare finne riktige variabler navn sanne har brukt

#print("task 12: Socio-economic cost for solution A:", socialeco_cost_A, "NOK")


#task 13 - Socio-economic cost for solution B

#socialeco_cost_B = c_inv_pv + sum(cens_B_NOK) samme her^
#print("task 13: Socio-economic cost for solution B:", socialeco_cost_B, "NOK")
