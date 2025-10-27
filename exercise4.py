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
analysis_horizon = 20
economic_lifetime = 40


asset_age_at_end = analysis_horizon #- 1
remaining_fraction =  1 - asset_age_at_end / economic_lifetime

# Residual value and its PV
residual_value = capex_NOK * remaining_fraction
PV_residual = residual_value / ((1 + r) ** analysis_horizon)

# Corrected PV
PV_corrected = PV_capex - PV_residual

print(f"Residual value (undiscounted, end of year {analysis_horizon}): {residual_value:,.0f} NOK")
print(f"PV(investment): {PV_capex:,.0f} NOK")
print(f"PV(residual): {PV_residual:,.0f} NOK")
print(f"Corrected PV (PV_inv - PV_res): {PV_corrected:,.0f} NOK")

#TASK 6
growth = 0.03
limit_with_bess = 5.0  # 4 + 1 MW

# smallest y with P_max*(1+g)^y > 5
y_cross = int(np.ceil(np.log(limit_with_bess / P_max) / np.log(1 + growth)))
#print(f"First year peak exceeds 5 MW: year {y_cross} -> reinforce at start of year {y_cross+1}")

capex = 759_408 * 20.0   # 15,188,160 NOK
r = 0.04
analysis_horizon = 20
lifetime = 40
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

# --- Baseline A: invest at start of year 2 (discount 1 year) ---
PV_capex_A = capex / (1 + r)**1
age_A = y_end - 1          # 19 years used by end of year 20
residual_A = capex * (1 - age_A / T_life)
PV_res_A = residual_A / (1 + r)**y_end
PV_corr_A = PV_capex_A - PV_res_A

# --- Alternative B: deferral; invest at start of year 10 (discount 9 years) ---
PV_capex_B = capex / (1 + r)**9
age_B = y_end - 9 - 0      # asset age from start of year 10 to end of year 20 = 11
residual_B = capex * (1 - age_B / T_life)
PV_res_B = residual_B / (1 + r)**y_end
PV_corr_B = PV_capex_B - PV_res_B

PV_reduction = PV_corr_A - PV_corr_B

#print(f"Corrected PV (A, invest start year 2): {PV_corr_A:,.2f} NOK")
#print(f"Corrected PV (B, invest start year 10): {PV_corr_B:,.2f} NOK")
#print(f"PV reduction due to deferral:          {PV_reduction:,.2f} NOK")

#print(PV_corr_B)
#print(PV_capex_B)

#TASK 7



