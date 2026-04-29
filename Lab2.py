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
plt.title('Distribution of Primary Energy Consumption')
plt.xlabel('Primary Energy Consumption (TWh)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df_subset['carbon_intensity_elec'], color='lightgreen')
plt.title('Boxplot of Carbon Intensity of Electricity')
plt.xlabel('Carbon Intensity (gCO2/kWh)')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_subset, x='gdp', y='primary_energy_consumption', alpha=0.5, color='coral')
plt.title('Relationship between GDP and Energy Consumption')
plt.xlabel('GDP')
plt.ylabel('Primary Energy Consumption (TWh)')
plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = df_subset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Key Energy Variables')
plt.show()

# =====================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
