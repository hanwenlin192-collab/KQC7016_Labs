import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 载入数据
# 确保文件名与你上传的完全一致
df = pd.read_csv('WorldEnergy .csv')

# 选取几个核心变量进行分析，使结果更聚焦
cols_of_interest = ['year', 'population', 'gdp', 'primary_energy_consumption', 'greenhouse_gas_emissions', 'carbon_intensity_elec']
df_subset = df[cols_of_interest].copy()

print("--- 数据概览 ---")
print(df_subset.info())

# 2. 缺失值处理 (特征工程)
# 简单策略：用每列的中位数填充缺失值，避免画图时报错
df_subset = df_subset.fillna(df_subset.median())

# 设置绘图风格
sns.set_theme(style="whitegrid")

# 3. 单变量分析 (Univariate Analysis) - 直方图
plt.figure(figsize=(8, 5))
sns.histplot(df_subset['primary_energy_consumption'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Primary Energy Consumption')
plt.xlabel('Primary Energy Consumption (TWh)')
plt.ylabel('Frequency')
plt.show()

# 4. 单变量分析 (Univariate Analysis) - 箱线图
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_subset['carbon_intensity_elec'], color='lightgreen')
plt.title('Boxplot of Carbon Intensity of Electricity')
plt.xlabel('Carbon Intensity (gCO2/kWh)')
plt.show()

# 5. 双变量分析 (Bivariate Analysis) - 散点图
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_subset, x='gdp', y='primary_energy_consumption', alpha=0.5, color='coral')
plt.title('Relationship between GDP and Energy Consumption')
plt.xlabel('GDP')
plt.ylabel('Primary Energy Consumption (TWh)')
plt.show()

# 6. 多变量分析 (Multivariate Analysis) - 相关性热力图
plt.figure(figsize=(8, 6))
# 计算核心变量之间的相关系数
correlation_matrix = df_subset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Key Energy Variables')
plt.show()
