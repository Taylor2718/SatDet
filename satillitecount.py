# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:30:02 2023

@author: nehay

3 hours, now with means
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
import datetime
import os
#from empca import empca

#import tweepy

col = ['black','magenta','red','orange','lime','green','cyan','blue']

ourbands = [(10000,10200),
            (10200,10400),
            (10400,10600),
            (10600,10800),
            (10800,10950),
            (10950,11200),
            (11200,11450),
            (11450,11700),
            (11700,11950),
            (11950,12100)]         

bands = [(10.705,10.945),
         (10.955,11.195),
         (11.205,11.445),
         (11.455,11.695),
         (11.705,11.945),
         (11.955,12.195),
         (12.205,12.445),
         (12.455,12.695)]     
                                            
#%%Reading data

start_datetime = datetime.datetime(2023, 12, 2, 16)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 2, 18)  # End of analysis
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
                #nu = nu - 100 # for satillites only!!!!
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
#%%Creating waterfall
specs = np.array(specs)
base = np.median(specs,axis=0)
#%%
df = pd.DataFrame({'dates': dates})

# Sort the DataFrame by dates
df.sort_values(by='dates', inplace=True)

# Define the frequency of the chunks (10 minutes)
chunk_frequency = pd.to_timedelta('5T')

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
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

protect_min = 10600
protext_max = 10800  

# Combine conditions to extract columns within the specified range
mask_protect = (nu > protect_min) & (nu < protext_max)
specs_protect = specs[:, mask_protect]
waterfalls_protect = waterfalls[:, mask_protect]
nu_protect = nu[mask_protect]
mean_waterfalls_protect = np.sum(waterfalls_protect,axis=1) - np.mean(waterfalls_protect,axis=1)
mean_specs_protect = np.mean(specs_protect,axis=0)

# mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
# mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)

maskprotect = mean_waterfalls_protect < -100
datesprotect = np.array(dates)
# Apply the mask to filter out unwanted indices
filtered_dates_protect = datesprotect#[~mask]
mean_waterfalls_protect[maskprotect] = -2
filtered_mean_waterfalls_protect = mean_waterfalls_protect #[~mask]

# mask2 = mean_waterfall_2 < -1
# #dates = np.array(dates)
# # Apply the mask to filter out unwanted indices
# filtered_dates2 = dates[~mask2]
# filtered_mean_waterfall2 = mean_waterfall_2[~mask2]

# plt.plot(filtered_dates_protect, filtered_mean_waterfalls_protect)
# plt.show()

# plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
# plt.show()
# plt.plot(filtered_dates2, filtered_mean_waterfall2)
# plt.show()
#%%
no_of_peaks_antenna = []


lower_1 = 11200
upper_1 = 11450 

# Combine conditions to extract columns within the specified range
mask_1 = (nu > lower_1) & (nu < upper_1)
specs_1 = specs[:, mask_1]
waterfalls_1 = waterfalls[:, mask_1]
nu_1 = nu[mask_1]
mean_waterfalls_1 = np.sum(waterfalls_1,axis=1) - np.mean(waterfalls_1,axis=1)
mean_specs_1 = np.mean(specs_1,axis=0)

# mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
# mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)

mask = mean_waterfalls_1 < -100
dates = np.array(dates)
# Apply the mask to filter out unwanted indices
filtered_dates = dates#[~mask]
mean_waterfalls_1[mask] = -2
filtered_mean_waterfalls = mean_waterfalls_1 #[~mask]

# mask2 = mean_waterfall_2 < -1
# #dates = np.array(dates)
# # Apply the mask to filter out unwanted indices
# filtered_dates2 = dates[~mask2]
# filtered_mean_waterfall2 = mean_waterfall_2[~mask2]

# plt.plot(filtered_dates, filtered_mean_waterfalls)
# plt.show()

max_noise = max(filtered_mean_waterfalls_protect)
peaks, _ = find_peaks(filtered_mean_waterfalls, height=max_noise+1, width=8, distance = 120)
waterfall_color = "#87CEEB"  # Light Sky Blue
peak_color = "#FF6347"  # Tomato
max_noise_color = "#A9A9A9"  # Dark Gray
plt.plot(filtered_dates, filtered_mean_waterfalls, label = "Waterfall Sum",alpha=0.5)
plt.plot(filtered_dates[peaks], filtered_mean_waterfalls[peaks], "x", label = "Peak", color='blue')
plt.hlines(max_noise, filtered_dates[0], filtered_dates[-1], linestyle='--', label='Max Noise Threshold', color='black')
plt.ylabel("Amplitude (dB)")
plt.xlabel("Time")
plt.legend()
plt.savefig("Figures/Peak-Detector.svg", format='svg', bbox_inches='tight')


#%%
for i in ourbands:
    
    lower_1 = i[0]
    upper_1 = i[1]  
    
    # Combine conditions to extract columns within the specified range
    mask_1 = (nu > lower_1) & (nu < upper_1)
    specs_1 = specs[:, mask_1]
    waterfalls_1 = waterfalls[:, mask_1]
    nu_1 = nu[mask_1]
    mean_waterfalls_1 = np.sum(waterfalls_1,axis=1) - np.mean(waterfalls_1,axis=1)
    mean_specs_1 = np.mean(specs_1,axis=0)
    
    # mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
    # mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)
    
    mask = mean_waterfalls_1 < -100
    dates = np.array(dates)
    # Apply the mask to filter out unwanted indices
    filtered_dates = dates#[~mask]
    mean_waterfalls_1[mask] = -2
    filtered_mean_waterfalls = mean_waterfalls_1 #[~mask]
    
    # mask2 = mean_waterfall_2 < -1
    # #dates = np.array(dates)
    # # Apply the mask to filter out unwanted indices
    # filtered_dates2 = dates[~mask2]
    # filtered_mean_waterfall2 = mean_waterfall_2[~mask2]
    
    # plt.plot(filtered_dates, filtered_mean_waterfalls)
    # plt.show()
    
    max_noise = max(filtered_mean_waterfalls_protect)
    peaks, _ = find_peaks(filtered_mean_waterfalls, height=max_noise+1, width=8, distance = 120)
    plt.plot(filtered_dates, filtered_mean_waterfalls)
    #plt.plot(filtered_dates, filtered_mean_waterfalls[peaks], "x")
    plt.plot(filtered_dates[peaks], filtered_mean_waterfalls[peaks], "x")
    #plt.plot(np.zeros_like(filtered_mean_waterfalls), "--", color="gray")
    plt.title(i)
    plt.show()
    
    print(len(peaks))
    no_of_peaks_antenna.append(len(peaks))
    mean_specs = np.mean(specs,axis=0)
    max_specs = np.max(specs,axis=0)
    min_specs = np.min(specs,axis=0)
    
    # plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
    # plt.show()
    # plt.plot(filtered_dates2, filtered_mean_waterfall2)
    # plt.show()
#%%
#%%Reading data

start_datetime = datetime.datetime(2023, 12, 1, 16)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 1, 18)  # End of analysis
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
                #nu = nu - 100 # for satillites only!!!!
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
#%%Creating waterfall
specs = np.array(specs)
base = np.median(specs,axis=0)
#%%
df = pd.DataFrame({'dates': dates})

# Sort the DataFrame by dates
df.sort_values(by='dates', inplace=True)

# Define the frequency of the chunks (10 minutes)
chunk_frequency = pd.to_timedelta('5T')

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
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

protect_min = 10600
protext_max = 10800  

# Combine conditions to extract columns within the specified range
mask_protect = (nu > protect_min) & (nu < protext_max)
specs_protect = specs[:, mask_protect]
waterfalls_protect = waterfalls[:, mask_protect]
nu_protect = nu[mask_protect]
mean_waterfalls_protect = np.sum(waterfalls_protect,axis=1) - np.mean(waterfalls_protect,axis=1)
mean_specs_protect = np.mean(specs_protect,axis=0)

# mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
# mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)

maskprotect = mean_waterfalls_protect < -100
datesprotect = np.array(dates)
# Apply the mask to filter out unwanted indices
filtered_dates_protect = datesprotect#[~mask]
mean_waterfalls_protect[maskprotect] = -2
filtered_mean_waterfalls_protect = mean_waterfalls_protect #[~mask]

# mask2 = mean_waterfall_2 < -1
# #dates = np.array(dates)
# # Apply the mask to filter out unwanted indices
# filtered_dates2 = dates[~mask2]
# filtered_mean_waterfall2 = mean_waterfall_2[~mask2]

# plt.plot(filtered_dates_protect, filtered_mean_waterfalls_protect)
# plt.show()

# plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
# plt.show()
# plt.plot(filtered_dates2, filtered_mean_waterfall2)
# plt.show()
#%%
no_of_peaks_satillite = []

for i in ourbands:
    
    lower_1 = i[0]
    upper_1 = i[1]  
    
    # Combine conditions to extract columns within the specified range
    mask_1 = (nu > lower_1) & (nu < upper_1)
    specs_1 = specs[:, mask_1]
    waterfalls_1 = waterfalls[:, mask_1]
    nu_1 = nu[mask_1]
    mean_waterfalls_1 = np.sum(waterfalls_1,axis=1) - np.mean(waterfalls_1,axis=1)
    mean_specs_1 = np.mean(specs_1,axis=0)
    
    # mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
    # mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)
    
    mask = mean_waterfalls_1 < -100
    dates = np.array(dates)
    # Apply the mask to filter out unwanted indices
    filtered_dates = dates#[~mask]
    mean_waterfalls_1[mask] = -2
    filtered_mean_waterfalls = mean_waterfalls_1 #[~mask]
    
    # mask2 = mean_waterfall_2 < -1
    # #dates = np.array(dates)
    # # Apply the mask to filter out unwanted indices
    # filtered_dates2 = dates[~mask2]
    # filtered_mean_waterfall2 = mean_waterfall_2[~mask2]
    
    # plt.plot(filtered_dates, filtered_mean_waterfalls)
    # plt.show()
    
    max_noise = max(filtered_mean_waterfalls_protect)
    peaks, _ = find_peaks(filtered_mean_waterfalls, height=max_noise+1, width=8, distance = 120)
    plt.plot(filtered_dates, filtered_mean_waterfalls)
    #plt.plot(filtered_dates, filtered_mean_waterfalls[peaks], "x")
    plt.plot(filtered_dates[peaks], filtered_mean_waterfalls[peaks], "x")
    #plt.plot(np.zeros_like(filtered_mean_waterfalls), "--", color="gray")
    plt.title(i)
    plt.show()
    
    print(len(peaks))
    no_of_peaks_satillite.append(len(peaks))
    mean_specs = np.mean(specs,axis=0)
    max_specs = np.max(specs,axis=0)
    min_specs = np.min(specs,axis=0)
    
    # plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
    # plt.show()
    # plt.plot(filtered_dates2, filtered_mean_waterfall2)
    # plt.show()
#%%
plt.plot(no_of_peaks_antenna, 'x-', label = 'Antenna')
plt.plot(no_of_peaks_satillite, 'x-', label = 'Satillite Dish')
plt.legend()
plt.title("Satillite Count (over 3 hours)")
plt.ylabel('Count')
plt.xlabel('Band')
#txt="Comparitive satillite count over 3 hours of using our" "3D printed antenna vs a satillite dish on consecutuve days, from 16:00 - 19:00"
#plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
