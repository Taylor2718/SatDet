#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:41:32 2023

@author: taylo
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import sem
import datetime
import os
#import tweepy

col = ['black','magenta','red','orange','lime','green','cyan','blue']

bands = [(10.705,10.945),
         (10.955,11.195),
         (11.205,11.445),
         (11.455,11.695),
         (11.705,11.945),
         (11.955,12.195),
         (12.205,12.445),
         (12.455,12.695)]                                                  
#%%Reading data

start_datetime = datetime.datetime(2023, 12, 2, 0)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 2, 1)  # End of analysis
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
    # Move to the next hour
    current_datetime += datetime.timedelta(hours=1)
    
print(f"You have successfully processed {file_count} files comprising of {len(dates)} scans ranging from {dates[0]} to {dates[-1]}")

# Function to format datetime object into the desired ID format
def format_id(date):
    return f"{date.year - 2000:02d}{date.month:02d}{date.day:02d}{date.hour:02d}{date.minute:02d}{date.second:02d}{int(date.microsecond / 1e5)}"

# Create a list of IDs
ids = [format_id(date) for date in dates]
#%%Creating waterfall
specs = np.array(specs)
base = np.median(specs,axis=0)
#%%

waterfall = specs-base

#%%Plotting waterfall
vmin = np.percentile(waterfall, 5)
vmax = np.percentile(waterfall, 95)

plt.figure(figsize=(10, 6))
plt.imshow(waterfall, aspect='auto', cmap='viridis',  vmin=vmin, vmax=vmax)

# Set x-axis ticks and labels based on the frequency array 'nu'
num_ticks = 6  # Adjust the number of ticks as needed

ticks_positions = np.linspace(0, len(nu) - 1, num_ticks)
ticks_labels = np.linspace(10000, 12000, num_ticks)

plt.xticks(ticks_positions, ticks_labels)

plt.colorbar(label='Amplitude')
plt.xlabel('Frequency')
plt.ylabel('Time')
plt.title('Waterfall Plot')
plt.show()

#%%Plotting mean waterfall (collapsing time)
mean_waterfall = np.mean(waterfall,axis=0)
sem_waterfall = sem(waterfall,axis=0)
max_waterfall = np.max(waterfall,axis=0)
min_waterfall = np.min(waterfall,axis=0)

plt.plot(nu,mean_waterfall, '--', color = 'black', label = 'Mean')
plt.plot(nu,min_waterfall, color = 'blue', label = 'Min Hold')
plt.plot(nu,max_waterfall, color = 'red', label = 'Max Hold')

#plt.fill_between(nu, mean_waterfall - sem_waterfall, mean_waterfall + sem_waterfall, color='blue', alpha=0.1, label='SEM')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative Amplitude (a.u.)")
plt.legend()
plt.grid()

#%%Plotting waterfall for bands (collapisng frequency)
lower_1 = 11650
upper_1 = 11800  

# Combine conditions to extract columns within the specified range
mask_1 = (nu > lower_1) & (nu < upper_1)
specs_1 = specs[:, mask_1]
waterfall_1 = waterfall[:, mask_1]
nu_1 = nu[mask_1]
mean_waterfall_1 = np.mean(waterfall_1,axis=1)
mean_specs_1 = np.mean(specs_1,axis=0)

mask = mean_waterfall_1 < -1
dates = np.array(dates)
# Apply the mask to filter out unwanted indices
filtered_dates = dates[~mask]
filtered_mean_waterfall = mean_waterfall_1[~mask]

plt.plot(filtered_dates, filtered_mean_waterfall)

#%%Plotting Fourier transform
from scipy.signal import hann

time_step = np.diff(filtered_dates).mean().total_seconds()
fft_result = np.fft.fft(filtered_mean_waterfall)

exp = np.log10(np.max(np.abs(fft_result)))/5
#fft_result_shifted = np.fft.fftshift(fft_result)
windowed_data = hann(len(filtered_mean_waterfall)) * filtered_mean_waterfall

# Calculate the frequency values
frequencies = np.fft.fftfreq(len(fft_result), d=time_step)
white_noise = np.ones_like(frequencies)
one_over_f_noise = 1 / (frequencies)**(1)
one_over_f_noise[0] = 0  # Avoid division by zero
# Plot the magnitude spectrum with a logarithmic scale for the frequency axis
plt.plot(frequencies, np.abs(fft_result), label='FFT')
plt.plot(frequencies, white_noise, label='White Noise', linestyle='--')
plt.plot(frequencies, one_over_f_noise, label='1/f Noise', linestyle='--')

plt.title('Fourier Transform of Relative Amplitude Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xscale('log')  # Use a logarithmic scale for the frequency axis
plt.yscale('log')  # Use a logarithmic scale for the frequency axis
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5)  # Add a grid with dashed lines
plt.legend()
plt.show()


