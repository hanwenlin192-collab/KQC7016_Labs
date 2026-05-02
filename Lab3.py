import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. Data Loading & Cleaning
df = pd.read_csv('WorldEnergy.csv')

# Use cross-sectional data from 2021 to ensure sample independence
df_2021 = df[df['year'] == 2021].copy()

# Filter out regional aggregates (keep only sovereign countries)
clean_data = df_2021.dropna(subset=['iso_code', 'gdp', 'energy_per_capita', 'population']).copy()
clean_data = clean_data[~clean_data['iso_code'].str.startswith('OWID')]

# 2. Feature Engineering (Creating Categorical Factors)
# Factor 1: GDP (3 levels: Low, Medium, High)
clean_data['GDP_Group'] = pd.qcut(clean_data['gdp'], q=3, labels=['Low', 'Medium', 'High'])
# Factor 2: Population (2 levels: Low_Pop, High_Pop)
clean_data['Pop_Group'] = pd.qcut(clean_data['population'], q=2, labels=['Low_Pop', 'High_Pop'])

# ==========================================
# PART 1: One-Way ANOVA (Effect of GDP Group)
# ==========================================
print("--- 1-Way ANOVA (Effect of GDP on Energy Per Capita) ---")
model_1way = ols('energy_per_capita ~ C(GDP_Group)', data=clean_data).fit()
anova_1way = sm.stats.anova_lm(model_1way, typ=2)
print(anova_1way)

# Tukey's HSD Post-hoc Test
print("\n--- Tukey HSD (Which GDP groups differ?) ---")
tukey_result = pairwise_tukeyhsd(endog=clean_data['energy_per_capita'], 
                                 groups=clean_data['GDP_Group'], 
                                 alpha=0.05)
print(tukey_result)

# Visualization 1: Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='GDP_Group', y='energy_per_capita', data=clean_data, palette='Set2')
plt.yscale('log')
plt.title('Energy Per Capita by GDP Group (Log Scale)')
plt.ylabel('Energy Per Capita (kWh)')
plt.show()

# ==========================================
# PART 2: Two-Way ANOVA (GDP × Population)
# ==========================================
print("\n--- 2-Way ANOVA (GDP & Population Interaction) ---")
model_2way = ols('energy_per_capita ~ C(GDP_Group) * C(Pop_Group)', data=clean_data).fit()
anova_2way = sm.stats.anova_lm(model_2way, typ=2)
print(anova_2way)

# Visualization 2: Interaction Plot
plt.figure(figsize=(8, 6))
sns.pointplot(x='GDP_Group', y='energy_per_capita', hue='Pop_Group', data=clean_data, 
              dodge=True, markers=['o', 's'], capsize=.1, palette='muted')
plt.title('Interaction Effect: GDP & Population on Energy Demand')
plt.ylabel('Energy Per Capita (kWh)')
plt.show()

# ==========================================
# PART 3: Assumption Checks (Normality & Variance)
# ==========================================
print("\n--- Assumptions Check ---")
residuals = model_2way.resid
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test (Normality): p-value = {shapiro_test.pvalue:.4e}")

group_low = clean_data[clean_data['GDP_Group'] == 'Low']['energy_per_capita']
group_high = clean_data[clean_data['GDP_Group'] == 'High']['energy_per_capita']
levene_test = stats.levene(group_low, group_high)
print(f"Levene's Test (Homogeneity of Variance): p-value = {levene_test.pvalue:.4e}")
