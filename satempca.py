#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:05:10 2023

@author: taylo
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
import datetime
import os
from empca_file import empca

#%%Reading data

start_datetime = datetime.datetime(2023, 12, 2, 0)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 2, 10)  # End of analysis
directory = "Transits" #Set to your local transits folder

def format_datetime_string_directory(dt):
    return f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}/{dt.hour:02d}"

def format_datetime_string(dt):
    return f"HackRF_{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}00.npy"

current_datetime = start_datetime
file_count = 0
nwaterfall = 0
dates = []
specs = []
while current_datetime <= end_datetime:
    # Generate the directory path based on the current date and hour
    directory_path = os.path.join(directory, format_datetime_string_directory(current_datetime))
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate through the files in the directory
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(directory_path, file_name)
                print("Succesfully processed", file_name)
                file_count += 1
                # Read the data from the file
                #data = np.load(file_path, allow_pickle=True)
                pf = open(file_path,'rb')
                w = np.load(pf,allow_pickle=True)

                lo = float(w[3])     # LO freq tagged on end
                nu = w[1]/1e6        # Frequency array in MHz
                spec = w[2]          # Band amplutudes in dB
                nu_start = nu[0]     # Keep start freq as reference 
                start = w[0] 

                ipos = np.argsort(nu)    # Reorder interstepped data
                nu = nu[ipos]
                spec = spec[ipos]
                nu_start = nu[0]     # Keep start freq as reference    
                nspec = len(spec)
                #wei = gen_weights(nspec,40,100)  # Generate weighting function

                nu_lo = nu[0] + 10782.72 #10410     # Lowest freq of low band
                nu_hi = nu[-1] + 10782.72 #10410   # Highest freq of high band
                nu_step = nu[1]-nu[0]    # Frequency step

                nsamp = int((nu_hi-nu_lo)/nu_step) # Array size for combined spec

                not_EOF = True
                
                while not_EOF:
                    
                    scan = np.zeros(nsamp+1)
                    hit = np.zeros(nsamp+1)
                    
                    for i in range(5):  # Combine 4 scans ~1s
                        try:
                            w = np.load(pf,allow_pickle=True)
                        except:
                            not_EOF = False
                            break

                        date = w[0]          # Store date for check
                        lo = float(w[3])     # LO freq tagged on end
                        nu = w[1]/1e6        # Freq array
                        spec = w[2]          # Amp array
                        ips = np.argsort(nu) # De-jitter data   
                        nu = nu[ips]
                        spec = spec[ips]
                       
                        ipos = ((lo + nu - nu_lo)/nu_step).astype(int) # position output arr
                        ip = ((nu-nu[0])/nu_step).astype(int)          # position input arr

                        hit[ipos] = hit[ipos] + 1 #wei[ip]                # Weighted sum
                        scan[ipos] = scan[ipos] + spec  #*wei[ip]

                        ipos = np.where(hit>0)[0]  # Just use data where there are hits
                        scan[ipos] /= hit[ipos]
                        
                        dates.append(date)   
                        specs.append(scan)         # Add spec
                    
                nu = nu + 9750
                specs = specs[:-1]
                dates = dates[:-1]
    # Move to the next hour
    current_datetime += datetime.timedelta(hours=1)
    
print(f"You have successfully processed {file_count} files comprising of {len(dates)} scans ranging from {dates[0]} to {dates[-1]}")

# Function to format datetime object into the desired ID format
def format_id(date):
    return f"{date.year - 2000:02d}{date.month:02d}{date.day:02d}{date.hour:02d}{date.minute:02d}{date.second:02d}{int(date.microsecond / 1e5)}"

# Create a list of IDs
ids = [format_id(date) for date in dates]

specs = np.array(specs)
base = np.median(specs,axis=0)

#%%Group
df = pd.DataFrame({'dates': dates})

# Sort the DataFrame by dates
df.sort_values(by='dates', inplace=True)

# Define the frequency of the chunks (10 minutes)
chunk_frequency = pd.to_timedelta('1T')

# Create a new column with the chunk labels
df['chunk'] = (df['dates'] - df['dates'].iloc[0]) // chunk_frequency

# Group by the chunks and get the indices for each chunk
groups = df.groupby('chunk').apply(lambda x: x.index.tolist())

spec_min = np.min(specs, axis=0)
waterfalls = []
waterfalls_max = []
for group_indices in groups:

    base = np.median(specs[group_indices], axis=0)

    waterfall = specs[group_indices] - base
    waterfall = np.vstack(waterfall)
    waterfall_max = np.max(waterfall,axis=0)
    waterfalls.append(waterfall)
    waterfalls_max.append(waterfall_max)
    
waterfalls = np.vstack(waterfalls)
waterfalls_max = np.vstack(waterfalls_max)

mean_waterfall = np.mean(waterfalls,axis=0)
sem_waterfall = sem(waterfalls,axis=0)
max_waterfall = np.max(waterfalls,axis=0)
min_waterfall = np.min(waterfalls,axis=0)

medians=np.median(waterfalls_max, axis=1) #normalising eigenspectra suitable for PCA
waterfalls_max = waterfalls_max / medians[:,np.newaxis]
#%%Perform EMPCA
empca_niter_size = 10 #10 iterations
empca_nvec_size = 6 #using 6 eigenspec
m = empca(waterfalls_max, niter=empca_niter_size, nvec=empca_nvec_size)

waterfall_empca = m.model
eigvec = m.eigvec
coeff = m.coeff

#%%Plotting EMPCA reconstruction
plt.plot(nu, waterfalls_max[10], label = "Data", color='black',alpha=0.2)
plt.plot(nu, waterfall_empca[10], label = "Linear EMPCA", color='blue', alpha=0.7)

#%%Plotting eigenspectra
num_rows = (empca_nvec_size + 1) // 2
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12))

if num_rows == 1:
    axes = [axes]
    

fig.text(0.5, 0.05, 'Frequency (MHz)', ha='center', va='center', fontsize=14)
fig.text(0.06, 0.5, 'Relative Ampltiude', ha='center', va='center', rotation='vertical', fontsize=14)

for i in range(empca_nvec_size):
    row_idx = i // num_cols
    col_idx = i % num_cols
    ax = axes[row_idx][col_idx]
    ax.plot(nu, eigvec[i], label=r"$\phi_{%d}$" % (i+1))
    ax.legend(loc='upper right')  # Place the legend in the top right corner
    ax.grid(True, linestyle='--', alpha=0.5)
    
#%%Plotting coefficients

coeff_size = coeff.shape[1]

fig, axes = plt.subplots(coeff_size, coeff_size + 1, figsize=(20, 20), 
                         gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

bin_size = int(np.round(np.sqrt(coeff.shape[0])))

for i in range(coeff_size):
    for j in range(i):
        
        ax=axes[i, j]
                
        scatter = ax.scatter(coeff[:, i], coeff[:, j],
                             marker='.', s=20, alpha = 0.5)
        ax.grid(True)

        if i - j == 1:
            ax.annotate(f"PCA {j+1}", xy=(0.5, 1.05), xycoords='axes fraction', 
                        ha='center', fontsize=10, color='black')
            ax.annotate(f"PCA {i+1}", xy=(1.1, 0.5), xycoords='axes fraction', 
                    ha='center', fontsize=10, color='black', rotation=270, va='center')
    
    # Remove ticks and labels for the empty subplots
    for j in range(i, coeff_size):
        axes[i, j].axis('off')

    # Plot eigenspectra components in the 10th column
    ax = axes[i, coeff_size] #coeff_size = 10th column
    ax.plot(nu, eigvec[i], color='black',alpha=0.5)
    ax.annotate(r"$\phi_{%d}$" % (i+1), xy=(1.1, 0.5), xycoords='axes fraction', 
                ha='center', fontsize=12, color='blue', va='center')
    ax.set_xlim(min(nu), max(nu))
    ax.set_ylim(min(eigvec[i]), max(eigvec[i]))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
# Rotate the colorbar tick labels to be horizontal
plt.show()