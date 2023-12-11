# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:30:58 2023

@author: nehay
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:29:07 2023

@author: nehay
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:30:02 2023

@author: nehay
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
import datetime
import os
import scipy
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

satillite_count_per_band_day1 = [] 
satillite_count_per_band_day2 = []       
satillite_count_per_band_day3 = []  
widths_day1 = []     
widths_day2 = [] 
widths_day3 = []                                                     
#%%Reading data

start_datetime = datetime.datetime(2023, 12, 2, 00)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 2, 23)  # End of analysis
directory = "Transits" #Set to your local transits folder

def format_datetime_string_directory(dt):
    return f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}/{dt.hour:02d}"

def format_datetime_string(dt):
    return f"HackRF_{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}00.npy"
#%%
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
datetimes = pd.DataFrame({'dates': dates})
datetimes["index"] = datetimes.index.tolist()

hourly_waterfalls = []

for i in range(24):
      df1 = datetimes[datetimes['dates'].dt.hour == i]
      hour_only_data = waterfalls[df1["index"]]
      hourly_waterfalls.append(hour_only_data)
      
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


for a in ourbands:
    
    satillite_count_per_hour = []
    for i in range(len(hourly_waterfalls)): 
        protect_min = 10600
        protext_max = 10800  
        
        hour_waterfalls = hourly_waterfalls[i]
        # Combine conditions to extract columns within the specified range
        mask_protect = (nu > protect_min) & (nu < protext_max)
        waterfalls_protect = hour_waterfalls[:, mask_protect]
        nu_protect = nu[mask_protect]
        mean_waterfalls_protect = np.sum(waterfalls_protect,axis=1) - np.mean(waterfalls_protect,axis=1)
        
        # mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
        # mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)
        
        filtered_mean_waterfalls_protect = mean_waterfalls_protect #[~mask]
        
        
        lower_1 = a[0]
        upper_1 = a[1]  
        
        # Combine conditions to extract columns within the specified range
        mask_1 = (nu > lower_1) & (nu < upper_1)
        waterfalls_1 = hour_waterfalls[:, mask_1]
        nu_1 = nu[mask_1]
        mean_waterfalls_1 = np.sum(waterfalls_1,axis=1) - np.mean(waterfalls_1,axis=1)
        
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
        widths = scipy.signal.peak_widths(filtered_mean_waterfalls, peaks, rel_height=0.5)
        widths_day1.append(widths)
        # plt.plot(filtered_dates, filtered_mean_waterfalls)
        # #plt.plot(filtered_dates, filtered_mean_waterfalls[peaks], "x")
        # plt.plot(filtered_dates[peaks], filtered_mean_waterfalls[peaks], "x")
        # #plt.plot(np.zeros_like(filtered_mean_waterfalls), "--", color="gray")
        # plt.title(i)
        # plt.show()
        
        # print(len(peaks))
        satillite_count_per_hour.append(len(peaks))
    
        
    satillite_count_per_band_day1.append(satillite_count_per_hour)
        # plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
        # plt.show()
        # plt.plot(filtered_dates2, filtered_mean_waterfall2)
        # plt.show()
#%%
start_datetime = datetime.datetime(2023, 12, 3, 00)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 3, 23)  # End of analysis
directory = "Transits" #Set to your local transits folder

def format_datetime_string_directory(dt):
    return f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}/{dt.hour:02d}"

def format_datetime_string(dt):
    return f"HackRF_{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}00.npy"
#%%
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
datetimes = pd.DataFrame({'dates': dates})
datetimes["index"] = datetimes.index.tolist()

hourly_waterfalls = []

for i in range(24):
      df1 = datetimes[datetimes['dates'].dt.hour == i]
      hour_only_data = waterfalls[df1["index"]]
      hourly_waterfalls.append(hour_only_data)
      
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

for a in ourbands:
    
    satillite_count_per_hour = []
    for i in range(len(hourly_waterfalls)): 
        protect_min = 10600
        protext_max = 10800  
        
        hour_waterfalls = hourly_waterfalls[i]
        # Combine conditions to extract columns within the specified range
        mask_protect = (nu > protect_min) & (nu < protext_max)
        waterfalls_protect = hour_waterfalls[:, mask_protect]
        nu_protect = nu[mask_protect]
        mean_waterfalls_protect = np.sum(waterfalls_protect,axis=1) - np.mean(waterfalls_protect,axis=1)
        
        # mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
        # mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)
        
        filtered_mean_waterfalls_protect = mean_waterfalls_protect #[~mask]
        
        
        
        lower_1 = a[0]
        upper_1 = a[1]  
        
        # Combine conditions to extract columns within the specified range
        mask_1 = (nu > lower_1) & (nu < upper_1)
        waterfalls_1 = hour_waterfalls[:, mask_1]
        nu_1 = nu[mask_1]
        mean_waterfalls_1 = np.sum(waterfalls_1,axis=1) - np.mean(waterfalls_1,axis=1)
        
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
        widths = scipy.signal.peak_widths(filtered_mean_waterfalls, peaks, rel_height=0.5)
        widths_day2.append(widths)
        # plt.plot(filtered_dates, filtered_mean_waterfalls)
        # #plt.plot(filtered_dates, filtered_mean_waterfalls[peaks], "x")
        # plt.plot(filtered_dates[peaks], filtered_mean_waterfalls[peaks], "x")
        # #plt.plot(np.zeros_like(filtered_mean_waterfalls), "--", color="gray")
        # plt.title(i)
        # plt.show()
        
        # print(len(peaks))
        satillite_count_per_hour.append(len(peaks))
        mean_specs = np.mean(specs,axis=0)
        max_specs = np.max(specs,axis=0)
        min_specs = np.min(specs,axis=0)
        
    satillite_count_per_band_day2.append(satillite_count_per_hour)
        # plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
        # plt.show()
        # plt.plot(filtered_dates2, filtered_mean_waterfall2)
        # plt.show()
#%%
start_datetime = datetime.datetime(2023, 12, 4, 00)  # Start of analysis
end_datetime = datetime.datetime(2023, 12, 4, 23)  # End of analysis
directory = "Transits" #Set to your local transits folder

def format_datetime_string_directory(dt):
    return f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}/{dt.hour:02d}"

def format_datetime_string(dt):
    return f"HackRF_{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}00.npy"
#%%
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
#%%
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
datetimes = pd.DataFrame({'dates': dates})
datetimes["index"] = datetimes.index.tolist()

hourly_waterfalls = []

for i in range(24):
      df1 = datetimes[datetimes['dates'].dt.hour == i]
      hour_only_data = waterfalls[df1["index"]]
      hourly_waterfalls.append(hour_only_data)
      
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

for a in ourbands:
    
    satillite_count_per_hour = []
    for i in range(len(hourly_waterfalls)): 
        protect_min = 10600
        protext_max = 10800  
        
        hour_waterfalls = hourly_waterfalls[i]
        # Combine conditions to extract columns within the specified range
        mask_protect = (nu > protect_min) & (nu < protext_max)
        waterfalls_protect = hour_waterfalls[:, mask_protect]
        nu_protect = nu[mask_protect]
        mean_waterfalls_protect = np.sum(waterfalls_protect,axis=1) - np.mean(waterfalls_protect,axis=1)
        
        # mean_waterfall_2 = np.sum(waterfall_1,axis=1) - np.mean(waterfall_1,axis=1)
        # mean_specs_2 = np.sum(specs_1,axis=0) - np.mean(specs_1,axis=0)
        
        filtered_mean_waterfalls_protect = mean_waterfalls_protect #[~mask]
        
        
        
        lower_1 = a[0]
        upper_1 = a[1]  
        
        # Combine conditions to extract columns within the specified range
        mask_1 = (nu > lower_1) & (nu < upper_1)
        waterfalls_1 = hour_waterfalls[:, mask_1]
        nu_1 = nu[mask_1]
        mean_waterfalls_1 = np.sum(waterfalls_1,axis=1) - np.mean(waterfalls_1,axis=1)
        
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
        widths = scipy.signal.peak_widths(filtered_mean_waterfalls, peaks, rel_height=0.5)
        widths_day3.append(widths)
        # plt.plot(filtered_dates, filtered_mean_waterfalls)
        # #plt.plot(filtered_dates, filtered_mean_waterfalls[peaks], "x")
        # plt.plot(filtered_dates[peaks], filtered_mean_waterfalls[peaks], "x")
        # #plt.plot(np.zeros_like(filtered_mean_waterfalls), "--", color="gray")
        # plt.title(i)
        # plt.show()
        
        # print(len(peaks))
        satillite_count_per_hour.append(len(peaks))
        mean_specs = np.mean(specs,axis=0)
        max_specs = np.max(specs,axis=0)
        min_specs = np.min(specs,axis=0)
        
    satillite_count_per_band_day3.append(satillite_count_per_hour)
        # plt.plot(filtered_dates[0:9300], filtered_mean_waterfall[0:9300])
        # plt.show()
        # plt.plot(filtered_dates2, filtered_mean_waterfall2)
        # plt.show()
#%%
band_no = list(range(10))
hours = list(range(24))

per_band = []
bars = []
for i in range(10):
    mean_occupancy = []
    yerror = []
    for a in range(24):
        mean_hour_occupancy = ((sum(widths_day1[a+i*24][0]) + sum(widths_day2[a+i*24][0]) + sum(widths_day3[a+i*24][0]))/len(hour_waterfalls))/3
        error = sem([sum(widths_day1[a+i*24][0])/len(hour_waterfalls),  sum(widths_day2[a+i*24][0])/len(hour_waterfalls), sum(widths_day3[a+i*24][0])/len(hour_waterfalls)])
        mean_occupancy.append(mean_hour_occupancy)
        yerror.append(error)
    per_band.append(mean_occupancy)
    bars.append(yerror)
#%%
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10, 8), sharex = True)
#fig.tight_layout()
#fig.subplots_adjust(bottom=0.95)#, right = 0.95)
ax = ax.flatten()
for i in range(len(per_band)):
    ax[i].plot(hours, per_band[i], color = 'blue', label = ourbands[i])
    ax[i].errorbar(hours, per_band[i], yerr = bars[i], ecolor = 'red', color = 'blue', capsize = 4)
    ax[i].legend(loc = 'upper left')
    #fig.suptitle("Intraday Profiles of Satillite Counts Per Band")
    #ax[i].set_xlabel("Hour")
    #ax[i].set_ylabel("Mean Satillite Count")
    fig.text(0.5, 0.04, 'Hour', ha='center')
    fig.text(0.04, 0.5, 'Hourly Mean Occupany Per Band', va='center', rotation='vertical')
    plt.legend()
plt.show()