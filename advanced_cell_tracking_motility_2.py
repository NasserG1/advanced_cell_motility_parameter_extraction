# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:23:20 2024

@author: Nasser
"""

#%% Imports and initial settings

# for compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function
import trackpy as tp
from scipy.optimize import curve_fit
from scipy.stats import t, linregress

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Update matplotlib parameters for consistent font usage
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Latin Modern Roman",
    "font.sans-serif": ["Helvetica"]
})
#    "font.family": "Latin Modern Roman",

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rc('figure', figsize=(8, 8))
plt.rc('image', cmap='gray')

# Experiment conditions
cell = r'Ac 30010 ' # cell line
cond = r'  '# extra important condition. i.e date control or 0.4% O2

# Define conditions constants linked to experiment
# mag = 1.634  # µm/PIXEL
# dt = 30  # SECONDS
betamax = 1.75
betamin = 0.55
betaminn = str(betamin)
betamaxx = str(betamax)
# filtpc = 16.1

# Define path
str_path = (r'E:\2023_2024_bulk\motility\_thesis\Fig4,3_a_b\230918_ac_motility_run1_dt45s')
path = Path(str_path)

def read_experiment_parameters(folder):
    params_path = Path(folder) / 'fitted_parameters.csv'
    params_df = pd.read_csv(params_path)
    dt = float(params_df[params_df['Parameter'] == 'dt (s)']['Value'])
    mag = float(params_df[params_df['Parameter'] == 'resolution (um/pix)']['Value'])
    return dt, mag

params = read_experiment_parameters(str_path)
dt = params[0]
mag= params[1]


im = pd.read_csv(path / '2unfiltered_individual_msds.csv')
imi = pd.read_csv(path / '2filtered_kept_msds.csv')
em = pd.read_csv(path / '2experimental_avg_msd.csv')
trajectories = pd.read_csv(path / '2filtered_trajectories.csv')

# Add these lines at the beginning of script to create the 'final' and 'segments' subfolders
path1 = path / 'plots'
path1.mkdir(parents=True, exist_ok=True)

#%% Get SEM error on individual msds

sigma1 = imi.iloc[:, 1:].std(axis=1) / np.sqrt(imi.iloc[:, 1:].count(axis=1))


#%% Plot MSDs

"""
A complete plot with: filtered iMSDs, kepts iMSDs, and the mean ensemble MSD 
"""

plt.rc('figure', figsize=(6, 5))
plt.rc('image', cmap='gray')

fig, ax = plt.subplots()

# Plot the data
ax.plot(im.iloc[:, 0] / 60, im, 'r', alpha=0.1)  # red lines, semitransparent
ax.plot(imi.iloc[:, 0] / 60, imi, 'c', alpha=0.3)  # cyan lines, semitransparent
ax.plot(em.iloc[:, 0] / 60, em.iloc[:, 1], 'k-', linewidth=3)  # black line, thick

# Set the title and labels with LaTeX formatting
title = r'MSD with ensemble mean'
ax.set_title(title, fontsize=16)

ax.set_xlabel(r'$t$ [min]', fontsize=16)
ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [µm²]', fontsize=16)

# Set font sizes for ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Set x and y axis scales to log
ax.set_xscale('log')
ax.set_yscale('log')

# Set x-axis limit to start where the data starts
min_x = min(im.iloc[:, 0].min(), imi.iloc[:, 0].min(), em.iloc[:, 0].min()) / 60
ax.set_xlim(left=min_x)

# Set y-axis limit to start where the data starts
min_y = min(im.min().min(), imi.min().min(), em.iloc[:, 1].min())
ax.set_ylim(bottom=min_y)
plt.tight_layout()

# Save the figure
fig.savefig(f"{path1}/eMSD.png", dpi=200)

# Show the plot
plt.show()


#%% Fit for D and P using classic Furth

"""
Fürth formula fitting to the MSD
A standard in biophysics for correlated stochastic walkers (as is the case for cellular motility)

The diffusion coefficient D and the persistence time P are extracted via fit.
 
"""


# Define the function to fit
def func(t, D, P):
    return 4*D*(t - P*(1 - np.exp(-t/P)))


xData = em.iloc[:, 0] / 60 # time in minutes
yData = em.iloc[:, 1] # ensemble MSD values

# Initial guess for the parameters
# Depends on the cellular organism
initialGuess = [1600, 5] #D and P

# Perform the curve-fit
popt, pcov = curve_fit(func, xData, yData, initialGuess, bounds=(0, np.inf))

# Statistics for 95% confidence interval
alpha = 0.05
n = len(yData)
p = len(initialGuess)
dof = max(0, n - p)
tval = t.ppf(1.0 - alpha / 2.0, dof)
sigma = np.sqrt(np.diag(pcov))
conf_interval = tval * sigma

plt.rc('figure', figsize=(6, 5))
plt.rc('image', cmap='gray')

# Plotting
fig, ax = plt.subplots()

ax.plot(im.iloc[:, 0]/60, im, 'r', alpha=0.1)
ax.plot(imi.iloc[:, 0]/60, imi, 'c', linewidth=3,alpha=0.3, zorder=1)
ax.errorbar(em.iloc[:, 0] / 60, em.iloc[:, 1], yerr=sigma1, fmt='k-', linewidth=3,alpha=0.7, 
            label=r'Exp. data: $\langle MSD_{{exp}} \rangle$', zorder=1)

# Plot the fitted function
ax.plot(xData, func(xData, *popt), 'r-', linewidth=3, label=rf"Fürth fit: $D={popt[0]:.2f} \, \mu\mathrm{{m}}^2/\mathrm{{min}}, \, \tau_p={popt[1]:.2f} \ \mathrm{{min}}$")
ax.legend(fontsize=12, loc='upper left')#, bbox_to_anchor=(0.05, 0.95))

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel(r'$t$ [min]', fontsize=20)
ax.set_ylabel(r'$\langle MSD \rangle$ [µm²]', fontsize=20)

ax.set_xscale('log') 
ax.set_yscale('log')

# Set x-axis limit to start where the data starts
min_x = min(im.iloc[:, 0].min(), imi.iloc[:, 0].min(), em.iloc[:, 0].min()) / 60
max_x = max(im.iloc[:, 0].max(), imi.iloc[:, 0].max(), em.iloc[:, 0].max()) / 60
ax.set_xlim(left=min_x, right=max_x)

# Set y-axis limit to start where the data starts
min_y = min(im.min().min(), imi.min().min(), em.iloc[:, 1].min())
ax.set_ylim(bottom=min_y)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=13, loc="upper left")  # Reverse the order of handles and labels


plt.tight_layout()

fig.savefig(f"{path1}/eMSD_with_fit.png", dpi=200)
plt.show()


#%% MSD over 4t 


"""
To validate the D fit, at long times persistence is lost by correlation dissipation
thus D can directly be computed by D_effective=MSD/4t
Plotting the Furth fit obtained independently in the previous step 
over the D_effective plot is an elegant way to reinforce the validity of the fit

"""


sigma_weights = imi.iloc[:, 1:].std(axis=1) / np.sqrt(imi.iloc[:, 1:].count(axis=1))

fig, ax = plt.subplots()

ax.errorbar(em.iloc[:, 0]/60, (em.iloc[:, 1])/(4*em.iloc[:, 0]/60), yerr=sigma_weights/(4*em.iloc[:, 0]/60), 
            fmt='k-', alpha=0.7, linewidth=3, label=r'$\langle MSD_{{exp}} \rangle$/4t',zorder=1)

ax.plot(xData, func(xData, *popt) / (4 * xData), 
        'r-',linewidth=3, label=r"$\langle MSD_{{fit}} \rangle$/4t")

# ax.set_title('Asymptotic Slope for Diffusion in MSD/4t with Furth fit', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel(r'$t$ [min]', fontsize=20)
ax.set_ylabel(r'$\langle MSD \rangle$/4t [µm²/min]', fontsize=20)

ax.set_xlim(left=0, right=max_x)
ax.set_ylim(bottom=0)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=13, loc="best")  # Reverse the order of handles and labels

# plt.grid()
plt.tight_layout()
fig.savefig(f"{path1}/MSD_4t_with_fit.png", dpi=200)
plt.show()


#%% VACF with error

"""
The velocity autocorrelation function, yet another approach to quantifying the persistence time
Persistence, or the correlation in directionality within cellular motility decays exponentially.
This fit models the loss of correlation over time with an exponential decay function and extracts P 
Used to cross reference the previous fits, again, as an elegant confirmation

The VACF can be obtained by the second derivative of the MSD/2.

"""

# Function to compute numerical second derivative with error propagation
def second_derivative_with_error(y, x, yerr):
    dydx = np.gradient(y, x)
    dydx_err = np.sqrt((np.gradient(yerr, x))**2 + (np.gradient(y, x))**2 * (yerr/x)**2)
    d2ydx2 = np.gradient(dydx, x)
    d2ydx2_err = np.sqrt((np.gradient(dydx_err, x))**2 + (np.gradient(dydx, x))**2 * (yerr/x)**2)
    return d2ydx2, d2ydx2_err

# Compute the second derivative and its error
time = em.iloc[:, 0] / 60  # Convert time to minutes
msd = em.iloc[:, 1]
msd_sem = sigma1

second_derivative, second_derivative_err = second_derivative_with_error(msd, time, msd_sem)

# Compute VACF and its error
vacf = second_derivative / 2
vacf_err = second_derivative_err / 2


# The 1st to 3rd points are usually noise arising from the 2nd derivative
# We can adjust to take into account the valid points 
# It is customary to normalize by the first valid point in VACF analysis
# i.e Correlation is perfect = 1 and decays with time
max_index = np.argmax(vacf)
vacf_normalized = vacf / vacf[3]
vacf_err_normalized = vacf_err / vacf[3]
time_normalized = time - time[3]

# Define the exponential decay function for fitting
def exp_decay(t, tau):
    return 1 * np.exp(-t / tau)

# Initial guess for the fit parameters
initial_guess = [20]

# Set bounds for the parameter to ensure they are within expected ranges
bounds = ([0], [np.inf])

# Perform the curve fit with bounds
popt, pcov = curve_fit(exp_decay, time_normalized, vacf_normalized, p0=initial_guess, bounds=bounds, sigma=vacf_err_normalized, absolute_sigma=True)

# Extract the fitted parameters
tau_fit = popt

# Save the VACF data and fitted parameters
vacf_data = pd.DataFrame({'t (min)': time_normalized, 'VACF': vacf_normalized})
vacf_data.to_csv(f"{path1}/vacf_data_normalized.csv", index=False)

fit_params = pd.DataFrame({'Tau': [tau_fit]})
fit_params.to_csv(f"{path1}/vacf_fit_params_single_exp.csv", index=False)

# Plot the VACF and the fitted curve
fig, ax = plt.subplots()

ax.errorbar(time_normalized, vacf_normalized,fmt='k.',linewidth=3, label='Exp. data') #yerr=vacf_err_normalized, fmt='k.-', label='Experimental data')
ax.plot(time_normalized, exp_decay(time_normalized, *popt), 'r-', linewidth=3, label=f"Fit:$\\tau_1={tau_fit:.2f}$ min")

ax.set_xlabel(r'$t$ [min]', fontsize=16)
ax.set_ylabel(r'$C_v^{{norm}}(t)$', fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax.set_xlim(left=0, right=60)
ax.set_ylim(bottom=-1, top=2)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=12, loc="upper left")  # Reverse the order of handles and labels

# plt.grid()

plt.tight_layout()
# Save the figure
fig.savefig(f"{path1}/VACF_with_fit.png", dpi=200)

plt.show()


#%% CME ANALYSIS

"""
The coefficient of movement efficiency

The total distance over the total path length of a cell over a window of time
This is yet another way to quantify persistence

+CME observed over a time window close the persistence time should 
    be close to 1 by definition
+CME observed over a time window larger than the persistence time should 
    be decreasing by definition

"""

def compute_cme(trajectory_segment):
    start_pos = np.array([trajectory_segment.iloc[0]['x'], trajectory_segment.iloc[0]['y']])
    end_pos = np.array([trajectory_segment.iloc[-1]['x'], trajectory_segment.iloc[-1]['y']])
    displacement = np.linalg.norm(end_pos - start_pos)
    
    # Calculate total path length
    path_length = 0
    for i in range(1, len(trajectory_segment)):
        current_pos = np.array([trajectory_segment.iloc[i]['x'], trajectory_segment.iloc[i]['y']])
        prev_pos = np.array([trajectory_segment.iloc[i-1]['x'], trajectory_segment.iloc[i-1]['y']])
        path_length += np.linalg.norm(current_pos - prev_pos)
    
    # Calculate CME
    cme = displacement / path_length if path_length > 0 else 0
    return cme

# Specify the total time t in minutes
total_time_minutes = 10  # Example: compute CME over 10 minutes
total_time_seconds = total_time_minutes * 60
time_steps = int(total_time_seconds / dt)

# Initialize a list to store CME data
cme_data = []

# Group by particle and compute CME for each valid segment of the trajectory
for particle, group in trajectories.groupby('particle'):
    num_segments = len(group) // time_steps
    for segment in range(num_segments):
        start_index = segment * time_steps
        end_index = start_index + time_steps
        trajectory_segment = group.iloc[start_index:end_index]
        cme = compute_cme(trajectory_segment)
        cme_data.append({'particle': particle, 'CME': cme, 'segment': segment + 1})

# Create a dataframe from the CME data
cme_df = pd.DataFrame(cme_data)

# Remove rows with CME value of 0
cme_df = cme_df[cme_df['CME'] != 0]

# Calculate mean and SEM of CME
cme_mean = cme_df['CME'].mean()
cme_sem = cme_df['CME'].sem()

# Create a summary dataframe
summary_df = pd.DataFrame({
    'CME Mean': [cme_mean],
    'CME SEM': [cme_sem],
    'Total Time (minutes)': [total_time_minutes]
})

# Save CME data and summary to an Excel file
output_path = path1 / 'CME_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    cme_df.to_excel(writer, sheet_name='CME Data', index=False)
    summary_df.to_excel(writer, sheet_name='CME Summary', index=False)

print(f"CME analysis complete. Results saved to {output_path}")


#%% Iterate over total time minutes from 1 to 20 minutes 

'''
iterate over 20 minutes (from 1 minute to 20 minutes)
store the resulting average CME and its SEM, along with the corresponding time into an excel sheet called CME_time
then plot and show the relation of how increasing time over which CME is computed affects the final average CME value 
'''

# Function to compute CME for a given trajectory segment over a specified time t in minutes
def compute_cme(trajectory_segment):
    start_pos = np.array([trajectory_segment.iloc[0]['x'], trajectory_segment.iloc[0]['y']])
    end_pos = np.array([trajectory_segment.iloc[-1]['x'], trajectory_segment.iloc[-1]['y']])
    displacement = np.linalg.norm(end_pos - start_pos)
    
    # Calculate total path length
    path_length = 0
    for i in range(1, len(trajectory_segment)):
        current_pos = np.array([trajectory_segment.iloc[i]['x'], trajectory_segment.iloc[i]['y']])
        prev_pos = np.array([trajectory_segment.iloc[i-1]['x'], trajectory_segment.iloc[i-1]['y']])
        path_length += np.linalg.norm(current_pos - prev_pos)
    
    # Calculate CME
    cme = displacement / path_length if path_length > 0 else 0
    return cme

# Initialize a list to store CME data over different times
cme_time_data = []

# Iterate over total time minutes from 1 to 20 minutes
for total_time_minutes in range(1, 21):
    total_time_seconds = total_time_minutes * 60
    time_steps = int(total_time_seconds / dt)
    
    # Initialize a list to store CME data for this specific time
    cme_data = []

    # Group by particle and compute CME for each valid segment of the trajectory
    for particle, group in trajectories.groupby('particle'):
        num_segments = len(group) // time_steps
        for segment in range(num_segments):
            start_index = segment * time_steps
            end_index = start_index + time_steps
            trajectory_segment = group.iloc[start_index:end_index]
            cme = compute_cme(trajectory_segment)
            if cme != 0:
                cme_data.append(cme)
    
    # Calculate mean and SEM of CME for this specific time
    if cme_data:
        cme_mean = np.mean(cme_data)
        cme_sem = np.std(cme_data) / np.sqrt(len(cme_data))
    else:
        cme_mean = np.nan
        cme_sem = np.nan
    
    # Store the results
    cme_time_data.append({'Total Time (minutes)': total_time_minutes, 'CME Mean': cme_mean, 'CME SEM': cme_sem})

# Create a dataframe from the CME time data
cme_time_df = pd.DataFrame(cme_time_data)

# Save CME time data to an Excel file
output_path = path1 / 'CME_time.xlsx'
cme_time_df.to_excel(output_path, index=False)

print(f"CME time analysis complete. Results saved to {output_path}")


#%% Plot the relation of how increasing time over which CME is computed affects the final average CME value

plt.figure(figsize=(10, 6))
plt.errorbar(cme_time_df['Total Time (minutes)'], cme_time_df['CME Mean'], yerr=cme_time_df['CME SEM'], fmt='-o', capsize=5)
plt.xlabel('Total Time (minutes)')
plt.ylabel('Average CME')
plt.title('Effect of Increasing Time on Average CME Value')
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_output_path = path1 / 'CME_vs_time.png'
plt.savefig(plot_output_path, dpi=200)

# Show the plot
plt.show()
