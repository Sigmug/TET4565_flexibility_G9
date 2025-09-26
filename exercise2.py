# -*- coding: utf-8 -*-
"""
Created on 2023-07-14

@author: ivespe

Intro script for Exercise 2 ("Load analysis to evaluate the need for flexibility") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""

# %% Dependencies

import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = '/Users/sannespakmo/Library/CloudStorage/OneDrive-Personal/Skole/9. semester/Fordypningsemne/Flexibility/Exercises/7703070'

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 0.637 

# Maximum load demand of new load being added to the system
P_max_new = 0.4

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

# %% Extract hourly load time series for a full year for all the load points in the CINELDI reference system
# (this code is made available for solving task 3)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Get all the days of the year
repr_days = list(range(1,366))

# Get normalized load profiles for representative days mapped to buses of the CINELDI reference grid;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve normalized load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*P_max_new

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])
# %%

pp.runpp(net)

# Task 1. plot the voltages at all buses
voltages_pu = net.res_bus['vm_pu']
voltages_subset = voltages_pu.loc[voltages_pu.index <= 96]
plt.figure(figsize=(10, 6))
plt.plot(voltages_subset.index, voltages_subset.values, marker='o', linestyle='-', label='Bus voltage (p.u.)')
plt.xlabel('Bus index')
plt.ylabel('Voltage magnitude [p.u.]')
plt.title('Bus voltage profile')
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
#plt.show()

voltage_profile = net.res_bus.vm_pu.loc[0:96]
lowest_voltage = min(voltage_profile)
lowest_voltage_bus = voltage_profile.idxmin()
print('The lowest voltage is ', lowest_voltage, ' p.u at bus ', lowest_voltage_bus)


#Task 2: Find how much the voltages decrease as the load demand in the area increases           

load_demand = net.load.loc[net.load['bus'].isin(bus_i_subset), 'p_mw'].values
print(load_demand)
aggregated_load = sum(load_demand)

load_table = pd.DataFrame({'Bus': bus_i_subset,
                           'Load demand [MW]': load_demand})
#load_table['Total'] = [aggregated_load]
#load_table = load_table.append({'Bus': 'Total', 'Load demand [MW]': aggregated_load}, ignore_index=True)
total_row = pd.DataFrame({'Bus': ['Total'], 'Load demand [MW]': [aggregated_load]})
load_table = pd.concat([load_table, total_row])
print(load_table)

#List to store results
scaling_factors = []
lowest_voltages = []
aggregated_loads = []

#Scaling the load demand
for scaling_factor in range(10, 21):
    scaling_factor /= 10
    net.load.loc[net.load['bus'].isin(bus_i_subset), 'p_mw'] = load_demand * scaling_factor
    pp.runpp(net)
    voltage_profile = net.res_bus.vm_pu.loc[0:96]
    lowest_voltage = min(voltage_profile)

    scaling_factors.append(scaling_factor)
    lowest_voltages.append(lowest_voltage)

for sf in scaling_factors:
    aggregated_loads.append(aggregated_load * sf)

#Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(aggregated_loads, lowest_voltages, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Lowest voltage as a function of the aggregated load demand in the area')
plt.xlabel('Aggregated Load Demand [MW]')
plt.ylabel('Lowest Voltage [p.u]')
plt.legend()
plt.grid(True)
#plt.savefig('TET4565_flexibility_project/Plots/Exercise2_Task2.a.pdf', format='pdf')
#plt.show()

#Task 3 and 4

#Load time series for the area
load_profiles = load_time_series_mapped.loc[:, bus_i_subset]

#Calculate the total load time series for the area
total_load_time_series = load_profiles.sum(axis=1)

max= max(total_load_time_series)
print('The maximum aggregated load in the area is ', max, ' MW')

max_load_bus = {}
for bus in load_profiles.columns:
    max_load_bus[bus] = load_profiles[bus].max()

print(max_load_bus)

#Plot the load time series for the area
plt.figure(figsize=(11, 4))
plt.plot(total_load_time_series.values, linewidth=1.0)
plt.title("Aggregated load demand time series (buses 90, 91, 92, 96)")
plt.xlabel("Hour of year")
plt.ylabel("Aggregated load [MW]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.show()

#Task 5

P_area_max = float(total_load_time_series.max())

ldc = np.sort(total_load_time_series.values)[::-1]          # descending
x_pct = np.linspace(0, 100, len(ldc), endpoint=False)       # % of hours

plt.figure(figsize=(10, 4.5))
plt.plot(x_pct, ldc, linewidth=1.5, label="Load duration curve")
# Optional reference lines:
plt.axhline(P_area_max, linestyle=":", label=f"Peak = {P_area_max:.3f} MW")
plt.axhline(0.637, linestyle="--", label="Cap 85–86 = 0.637 MW")  # if relevant

plt.title("Load duration curve – aggregated area load (buses 90, 91, 92, 96)")
plt.xlabel("Percentage of hours in the year [%]")
plt.ylabel("Aggregated load [MW]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.savefig("Images/area_load_duration_curve.png", dpi=200)
plt.show()

##testddddd