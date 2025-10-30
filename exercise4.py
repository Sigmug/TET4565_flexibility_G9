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
import seaborn as sns



# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
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

#print(f"Current peak (year 1): {peak_by_year.iloc[0]:.3f} MW")
#print(f"Peak in year 2:        {peak_by_year.iloc[1]:.3f} MW")
#if pd.notna(first_violation_year):
#    print(f"Constraint (4 MW) is first violated in year {first_violation_year}.")
#else:
#    print("Constraint not violated within the chosen horizon.")

# Quick plot
# Step plot (piecewise-constant by year)
plt.figure()

# Extend one extra year on x so the last step is visible
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
#plt.show()

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
#print(f"First year peak exceeds 5 MW: year {y_cross} -> reinforce at start of year {y_cross+1}")
# Simple step plot: peak growth, 5 MW effective limit, and congestion year marker

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
#plt.show()

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

# ===== TASK 7 (enkel versjon) =====
# Forutsetninger
cost_per_mwh = 2000.0     # NOK/MWh
days_per_year = 20        # antall "like" dager i året
P_limit = 4.0             # MW (nettets grense)
P_bess = 1.0              # MW (maks timesvis lastforskyvning fra batteriet)
growth = 0.03             # årlig vekst
planning_horizon = y_end  # f.eks. 20 år, gjenbruker variabel fra tidligere

# 1) Finn når representativ topp (denne dagen) først går over 5 MW
P_max_base = float(load_time_series_subset_aggr.max())      # MW på representativ dag (år 1)
reinforce_year = None
for y in range(1, planning_horizon + 1):
    peak_y = P_max_base * ((1 + growth) ** (y - 1))
    if peak_y > (P_limit + P_bess):   # overskrider 5 MW
        reinforce_year = y      # forsterker ved starten av neste år
        break

# 2) Beregn årlige kostnader (0 etter forsterkning)
annual_costs = {}
for y in range(1, planning_horizon + 1):
    # Hvis forsterket før eller ved starten av dette året -> ingen innkjøp
    if (reinforce_year is not None) and (y >= reinforce_year):
        annual_costs[y] = 0.0
        continue

    # Skaler hele døgnprofilen for dette året
    growth_factor = (1 + growth) ** (y - 1)
    load_y = load_time_series_subset_aggr * growth_factor  # MW per time

    # Energi som må flyttes: overskudd over 4 MW, avgrenset av 1 MW batteri
    excess = np.maximum(load_y - P_limit, 0.0)             # MW per time
    shifted_per_hour = np.minimum(excess, P_bess)          # MW per time (cap 1 MW)
    E_shift_day = float(shifted_per_hour.sum())            # MWh for hele dagen

    # Årskostnad = MWh per dag * pris * antall slike dager
    annual_costs[y] = E_shift_day * cost_per_mwh * days_per_year

# 3) Utskrift (kort og ryddig)
print("TASK 7: Yearly operating costs for congestion management (NOK/year)")
if reinforce_year is None:
    print("- No reinforcement in the horizon; services are purchased every year.")
else:
    print(f"- First year with peak > 5 MW: year {reinforce_year-1}.")
    print(f"- Reinforcement at the start of year {reinforce_year} (no purchases from this year onwards).")

for y in range(1, reinforce_year + 1):
    print(f"Year {y:2d}: {annual_costs[y]:,.0f} NOK")

#Task 8

# ==== EENS for Alternative A ====

# Inputs you already have
length_km      = 20.0      # main feeder length
P_avg_year1    = 1.841     # MW (average load in year 1)
growth         = 0.03      # 3% annual growth
planning_horizon = 10   # e.g., 20

# --- Fill these two from your reliability table (permanent faults only) ---
lambda_perm_per_km_year = 0.0397  # <-- set: faults per km per year (e.g., 0.05)
repair_time_hours       = 3  # <-- set: hours per fault (e.g., 8.0)

# Expected number of permanent faults per year on the main feeder
faults_per_year = lambda_perm_per_km_year * length_km

# EENS per year
print("\nTASK 8: Expected Energy Not Supplied (EENS) per year (MWh)")
eens = []
for y in range(1, planning_horizon + 1):
    P_avg_y = P_avg_year1 * ((1 + growth) ** (y - 1))   # MW
    EENS_y  = P_avg_y * faults_per_year * repair_time_hours  # MWh/year
    eens.append(EENS_y)
    print(f"Year {y:2d}: EENS = {EENS_y:.2f} MWh")

# Task 9
