#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:31:43 2023

@author: taylo
"""
import numpy as np  
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt

#%%Current Satellite pie chart

sat_lab = ['Localisation: 143', 'Communication: 5870', 'Earth Observation: 192', 'Space Observation: 78']
sat_count = [143, 5870, 192, 78]

colors = ['#FF82A1', 'skyblue', '#8CFDAB', '#B982C6']

fig, ax = plt.subplots()

## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
explode = (0, 0.1, 0, 0)  

wedges, texts, autotexts = ax.pie(sat_count, autopct='%1.1f%%', startangle=0, colors=colors, 
       explode = explode)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')

# Set title
ax.legend(wedges, sat_lab, title='Active Satellite Count')
# Save as SVG with high quality
plt.tight_layout()

plt.savefig('Figures/satellite_types.svg', format='svg', dpi=300,
            bbox_inches='tight', pad_inches=0.1)
# Show the plot (optional)
plt.show()

#%%Log Bar Chart for Filling
sat_lab = ['SG1, SGA2, SG2', 'OW, OW2', 'KP', 'XW', '$GW^{*}$', 'YNH', '$HWH^{*}$', 'LYNK',
           '$Astra ^{*}$', '$TEL^{*}$', '$HVNET^{*}$', '$SPINL^{*}$', '$GLOB^{*}$', '$SEM^{*}$', 'ESP']

sat_count = [34396, 7088, 3232, 966, 12992, 1000, 2000, 2000, 13620, 300, 1440, 1190, 3080,
             116640, 337323]

current_count = [5206, 634, 2, 12, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 3]

sat_count = np.array(sat_count)
sat_count = sat_count

plt.figure(figsize=(10, 6))
plt.bar(range(len(sat_lab)), sat_count, color='skyblue', label='Number of Satellite Fillings')
plt.bar(range(len(sat_lab)), current_count, color='blue', label='Current Number of Satellites')


plt.xlabel('Satellite Constellation', fontsize=12)
plt.ylabel(r'Number of Satellites ($\log_{10}$)', fontsize=12)
plt.xticks(range(len(sat_lab)), sat_lab, rotation=45, ha='right')
current_count = 5870
plt.axhline(y=current_count, color='red', linestyle='--', label='Current Satellite Communications Count (N=5870)')
plt.yscale('log')  # Use a logarithmic scale for the frequency axis
# Show the plot
plt.legend()

plt.savefig('Figures/constellation_filings.svg', format='svg', dpi=300,
            bbox_inches='tight', pad_inches=0.1)
plt.tight_layout()
plt.show()