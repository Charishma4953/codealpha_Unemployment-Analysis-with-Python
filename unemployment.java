# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load Dataset
# Replace 'unemployment.csv' with your dataset file
df = pd.read_csv('unemployment.csv')

# Step 2: Inspect Dataset
print("\n🔍 Dataset Head:\n", df.head())
print("\n📄 Dataset Info:")
print(df.info())
print("\n🧹 Missing Values:\n", df.isnull().sum())

# Step 3: Rename Columns for Simplicity
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Expecting: ['date', 'region', 'estimated_unemployment_rate']
# Adjust column names below to match your dataset if needed

# Step 4: Parse Date Column
df['date'] = pd.to_datetime(df['date'])

# Step 5: Drop Missing Values (if any)
df.dropna(subset=['estimated_unemployment_rate'], inplace=True)

# Step 6: Sort by Date
df = df.sort_values(by='date')

# Step 7: Plot Unemployment Rate Over Time by Region
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='estimated_unemployment_rate', hue='region', marker='o')
plt.title('📈 Unemployment Rate Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

# Step 8: COVID-19 Impact Analysis
# Assuming COVID-19 impact starts in March 2020
covid_start = pd.to_datetime('2020-03-01')
covid_end = pd.to_datetime('2021-12-31')
covid_period = df[(df['date'] >= covid_start) & (df['date'] <= covid_end)]

print("\n🦠 COVID Period Summary Stats by Region:")
print(covid_period.groupby('region')['estimated_unemployment_rate'].describe())

# Step 9: Compare Pre-COVID and During-COVID Averages
pre_covid = df[df['date'] < covid_start]['estimated_unemployment_rate'].mean()
during_covid = covid_period['estimated_unemployment_rate'].mean()
print(f"\n📊 Pre-COVID Avg Unemployment Rate: {pre_covid:.2f}%")
print(f"📊 During-COVID Avg Unemployment Rate: {during_covid:.2f}%")

# Step 10: Optional - Seasonal Decomposition (for 1 region)
single_region = df['region'].unique()[0]  # Pick first region, or specify
region_df = df[df['region'] == single_region].set_index('date')
region_ts = region_df['estimated_unemployment_rate'].asfreq('MS')

decomposition = seasonal_decompose(region_ts, model='additive', period=12)
decomposition.plot()
plt.suptitle(f'Seasonal Decomposition: {single_region}', fontsize=14)
plt.tight_layout()
plt.show()
