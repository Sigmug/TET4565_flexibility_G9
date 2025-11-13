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
#path_data_set         = 'C:\\Users\\graff\\OneDrive\\Dokumenter\\CINELDI_MV_reference_system_v_2023-03-06' 

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
#plt.show()

##testddddd
#Task 6
original_total_load = total_load_time_series.sum()
original_peak_load = total_load_time_series.max()
original_utilization_time = original_total_load / original_peak_load
print('Total load: ', original_total_load, 'MWh')
print('Peak load: ', original_peak_load, 'MW')
print('Utilization time: ', original_utilization_time, 'h')

original_peak_load_dict = {}
for bus in load_profiles.columns:
    original_peak_load_dict[bus] = load_profiles[bus].max()
sum_peak_loads = sum(original_peak_load_dict.values())
original_coincidence_factor = original_peak_load / sum_peak_loads
print('Coincidence factor: ', original_coincidence_factor)


#Task 7 - see overleaf

#Task 8

#new demand or new customer
new_load_profiles_vary = load_profiles.copy()
new_load_profiles_vary['New customer'] = new_load_time_series

new_total_load_time_series = new_load_profiles_vary.sum(axis=1)

new_total_load_time_series_sorted = new_total_load_time_series.sort_values(ascending=False).reset_index(drop=True)


plt.figure(figsize=(10,6))
plt.plot(new_total_load_time_series_sorted)
plt.title('Duration curve')
plt.xlabel('Hours')
plt.ylabel('Load demand [MW]')
plt.grid(True)
#plt.show()

#Task 9
new_peak_load = new_total_load_time_series.max()
print('The new peak load is ', new_peak_load, ' MW')
maximum_overloading = P_lim - new_peak_load
print('The maximum overloading is ', maximum_overloading, ' MW')
#print(new_load_time_series) - this is only for bus 90

#Task 10
hours_overloading = []
for demand in new_total_load_time_series:
    if demand > P_lim:
        hours_overloading.append(demand)
number_of_h_overloading = len(hours_overloading)
print('The number of hours with overloading is ', number_of_h_overloading, 'h')




#Task 13

def ldc(series):
    """Return load duration curve (descending)."""
    return np.sort(series.values)[::-1]

def overlimit_hours(series, P_lim):
    return int((series > P_lim).sum())

def overlimit_energy(series, P_lim):
    # MWh with 1-hour resolution
    return float((series - P_lim).clip(lower=0.0).sum())

def util_time(series):
    # MWh / MW = h
    return float(series.sum() / series.max())

# (c) Existing loads only (Task 5)
area_existing = total_load_time_series  # already computed in your code

# (b) Time-dependent new load (Task 7)
area_time_dep = new_total_load_time_series  # already computed in your code

# (a) Constant new load: +0.4 MW every hour
const_new = 0.4  # MW
area_const = area_existing + const_new

# Build LDCs
ldc_existing = ldc(area_existing)
ldc_time_dep = ldc(area_time_dep)
ldc_const    = ldc(area_const)
x_pct = np.linspace(0, 100, len(ldc_existing), endpoint=False)

# Plot
plt.figure(figsize=(10,5))
plt.plot(x_pct, ldc_existing, linewidth=1.2, label="(c) Existing only")
plt.plot(x_pct, ldc_time_dep, linewidth=1.2, label="(b) Time-dependent + new")
plt.plot(x_pct, ldc_const,    linewidth=1.2, label="(a) Constant 0.4 MW + new")
plt.axhline(P_lim, linestyle="--", label=f"P_lim = {P_lim:.3f} MW")
plt.xlabel("Percentage of hours in the year [%]")
plt.ylabel("Aggregated load [MW]")
plt.title("Load duration curves: constant vs. time-dependent new load vs. existing only")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.show()

# Key metrics table
rows = []
for name, series in [
    ("(c) Existing only", area_existing),
    ("(b) Time-dependent + new", area_time_dep),
    ("(a) Constant 0.4 MW + new", area_const),
]:
    rows.append({
        "Case": name,
        "Peak [MW]": float(series.max()),
        "Hours > P_lim [h]": overlimit_hours(series, P_lim),
        "Over-limit energy [MWh]": overlimit_energy(series, P_lim),
        "Utilization time [h]": util_time(series),
        "Annual energy [MWh]": float(series.sum()),
    })

df_q13 = pd.DataFrame(rows, columns=[
    "Case","Peak [MW]","Hours > P_lim [h]","Over-limit energy [MWh]",
    "Utilization time [h]","Annual energy [MWh]"
])
print("\n[Q13] LDC comparison metrics:")
print(df_q13.to_string(index=False))


#Task 14

# Helper metrics
def utilization_time(series):
    """MWh / MW = hours (hourly resolution)"""
    return float(series.sum() / series.max())

def coincidence_factor(area_series, per_bus_df):
    """
    CF = area peak / sum of individual peaks of contributing loads.
    per_bus_df columns = each load (buses, and optionally 'New').
    """
    per_bus_peaks = per_bus_df.max(axis=0).sum()
    return float(area_series.max() / per_bus_peaks)

# (c) Existing loads only (Task 5)
per_bus_existing = load_time_series_mapped.loc[:, bus_i_subset].copy()  # MW per bus
area_existing = per_bus_existing.sum(axis=1)

# (b) Time-dependent new load (Task 7)
per_bus_time_dep = per_bus_existing.copy()
per_bus_time_dep["New"] = new_load_time_series  # MW, time-dependent
area_time_dep = per_bus_time_dep.sum(axis=1)

# (a) Constant 0.4 MW new load (every hour)
const_new_value = 0.4  # MW
const_new_series = pd.Series(const_new_value, index=area_existing.index, name="New")
per_bus_const = per_bus_existing.copy()
per_bus_const["New"] = const_new_series
area_const = per_bus_const.sum(axis=1)

# Build summary table
rows = []
for name, area, per_bus in [
    ("(a) Existing + constant 0.4 MW", area_const, per_bus_const),
    ("(b) Existing + time-dependent",   area_time_dep, per_bus_time_dep),
    ("(c) Existing only",               area_existing, per_bus_existing),
]:
    rows.append({
        "Case": name,
        "Peak [MW]": float(area.max()),
        "Utilization time [h]": utilization_time(area),
        "Coincidence factor [-]": coincidence_factor(area, per_bus),
    })

df_q14 = pd.DataFrame(rows, columns=[
    "Case","Peak [MW]","Utilization time [h]","Coincidence factor [-]"
])

print("\n[Q14] Utilization time and coincidence factor:")
print(df_q14.to_string(index=False))

# Export a LaTeX table you can \input{} in the report
os.makedirs("tables", exist_ok=True)
latex_table = df_q14.to_latex(index=False, float_format="%.3f",
                              column_format="lccc",
                              caption="Utilization time and coincidence factor for cases (a)--(c).",
                              label="tab:q14-util-cf")
# Strip the outer table environment since we \input{} it inside a table in LaTeX
begin = latex_table.find("\\begin{tabular}")
end   = latex_table.find("\\end{tabular}") + len("\\end{tabular}")
with open("tables/q14_util_cf.tex", "w") as f:
    f.write(latex_table[begin:end])

print("\n[Q14] LaTeX saved to: tables/q14_util_cf.tex")



# Check that df_q14 exists; if not, raise a helpful error
if 'df_q14' not in globals():
    raise RuntimeError("df_q14 is missing. Run the Q14 code that computes df_q14 first.")

# Ensure a consistent order (a), (b), (c)
order = ["(a) Existing + constant 0.4 MW",
         "(b) Existing + time-dependent",
         "(c) Existing only"]
df_plot = df_q14.set_index("Case").loc[order].reset_index()

# Data
cases = df_plot["Case"].tolist()
util_time_vals = df_plot["Utilization time [h]"].values
cf_vals = df_plot["Coincidence factor [-]"].values

# Plot 1: Utilization time (h)
plt.figure(figsize=(8,4))
bars = plt.bar(cases, util_time_vals)
for b, v in zip(bars, util_time_vals):
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.0f} h",
             ha="center", va="bottom", fontsize=9)
plt.ylabel("Utilization time [h]")
plt.title("Utilization time for cases (a)–(c)")
plt.xticks(rotation=10)
plt.tight_layout()
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/q14_util_time.pdf", bbox_inches="tight")
# plt.savefig("figs/q14_util_time.png", dpi=300, bbox_inches="tight")
# plt.show()

# Plot 2: Coincidence factor (-)
plt.figure(figsize=(8,4))
bars = plt.bar(cases, cf_vals)
for b, v in zip(bars, cf_vals):
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)
plt.ylabel("Coincidence factor [-]")
plt.title("Coincidence factor for cases (a)–(c)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("figs/q14_coincidence_factor.pdf", bbox_inches="tight")
plt.savefig("figs/q14_coincidence_factor.png", dpi=300, bbox_inches="tight")
#plt.show()


#Task 15 - see overleaf
def pflex_cap(series, P_lim):
    return float((series - P_lim).clip(lower=0.0).max())

Pflex_a = pflex_cap(area_const, P_lim)      # Case (a): existing + constant 0.4 MW
Pflex_b = pflex_cap(area_time_dep, P_lim)   # Case (b): existing + time-dependent

print(f"[Q15] P_flex,cap (a): {Pflex_a:.3f} MW")
print(f"[Q15] P_flex,cap (b): {Pflex_b:.3f} MW")
