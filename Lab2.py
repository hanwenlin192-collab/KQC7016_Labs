import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('WorldEnergy.csv')

cols_of_interest = ['year', 'population', 'gdp', 'primary_energy_consumption', 'greenhouse_gas_emissions', 'carbon_intensity_elec']
df_subset = df[cols_of_interest].copy()

print("--- Data Overview ---")
print(df_subset.info())

df_subset = df_subset.fillna(df_subset.median())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df_subset['primary_energy_consumption'], bins=50, kde=True, color='skyblue')
plt.yscale('log')
plt.ylabel('Frequency (Log Scale)')
plt.title('Distribution of Primary Energy Consumption')
plt.xlabel('Primary Energy Consumption (TWh)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
real_data = df['carbon_intensity_elec'].dropna()
sns.violinplot(x=real_data, color='lightgreen', inner='quartile')
plt.title('True Distribution of Carbon Intensity (Missing Values Excluded)')
plt.xlabel('Carbon Intensity (gCO2/kWh)')
plt.show()

df = pd.read_csv('WorldEnergy.csv')
clean_scatter_data = df.dropna(subset=['gdp', 'primary_energy_consumption']).copy()

clean_scatter_data = clean_scatter_data[
    (clean_scatter_data['gdp'] > 0) & 
    (clean_scatter_data['primary_energy_consumption'] > 0)
]

clean_scatter_data['log_gdp'] = np.log10(clean_scatter_data['gdp'])
clean_scatter_data['log_energy'] = np.log10(clean_scatter_data['primary_energy_consumption'])

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.regplot(
    data=clean_scatter_data, 
    x='log_gdp', 
    y='log_energy', 
    scatter_kws={'alpha': 0.3, 'color': 'coral', 's': 20}, 
    line_kws={'color': 'darkred', 'linewidth': 2}
)

plt.title('Relationship between GDP and Energy Consumption (Log-Log Space)', fontsize=14)
plt.xlabel('Log10 (GDP)', fontsize=12)
plt.ylabel('Log10 (Primary Energy Consumption)', fontsize=12)

plt.show()

df = pd.read_csv('WorldEnergy.csv')

cols_of_interest = ['year', 'population', 'gdp', 'primary_energy_consumption', 'greenhouse_gas_emissions', 'carbon_intensity_elec']


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8)) 

sns.heatmap(
    real_correlation, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    vmin=-1, 
    vmax=1
)

plt.title('True Correlation Matrix of Key Energy Variables')
plt.show()

df = pd.read_csv('WorldEnergy.csv')

global_trend = df.groupby('year')['primary_energy_consumption'].sum(min_count=1).reset_index()

global_trend = global_trend.dropna(subset=['primary_energy_consumption'])
global_trend = global_trend[global_trend['primary_energy_consumption'] > 0]

global_trend = global_trend[global_trend['year'] <= 2023]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=global_trend, x='year', y='primary_energy_consumption', color='purple', linewidth=2.5)

plt.title('Global Primary Energy Consumption Trend (Up to 2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Primary Energy Consumption (TWh)', fontsize=12)

plt.fill_between(global_trend['year'], global_trend['primary_energy_consumption'], color='purple', alpha=0.1)

plt.xlim(global_trend['year'].min(), 2023)

plt.show()
