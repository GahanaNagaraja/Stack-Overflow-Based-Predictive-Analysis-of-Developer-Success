import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("survey_results_public.csv")

# Show top rows
df.head()

eda_columns = [
    'Age', 'EdLevel', 'Employment', 'RemoteWork',
    'ConvertedCompYearly', 'JobSat', 'YearsCodePro', 'LanguageHaveWorkedWith'
]

eda_df = df[eda_columns].copy()

# Missing values
missing = eda_df.isnull().sum().sort_values(ascending=False).reset_index()
missing.columns = ['Feature', 'MissingCount']

plt.figure(figsize=(10, 6))
sns.barplot(
    data=missing,
    x='MissingCount',
    y='Feature',
    hue='Feature',
    palette=sns.color_palette("Set2", n_colors=len(missing)),
    legend=False
)
plt.title("Missing Values")
plt.xlabel("Number of missing entries")
plt.ylabel("Feature")
plt.show()

# Salary distribution
plt.figure(figsize=(8, 4))
sns.histplot(eda_df['ConvertedCompYearly'], bins=50, kde=True)
plt.xlim(0, 300000)
plt.title("Salary Distribution (Clipped at $300K for Clarity)")
plt.xlabel("Salary in USD")
plt.show()

# Summary statistics
eda_df.describe(include='all')


# Drop rows missing key values
clean_df = eda_df.dropna(subset=['ConvertedCompYearly', 'YearsCodePro', 'EdLevel', 'Employment']).copy()

# Convert YearsCodePro to numeric safely
def convert_years_code(val):
    if val == 'Less than 1 year':
        return 0.5
    elif val == 'More than 50 years':
        return 51
    try:
        return float(val)
    except:
        return None

clean_df.loc[:, 'YearsCodePro'] = clean_df['YearsCodePro'].apply(convert_years_code)
clean_df = clean_df.dropna(subset=['YearsCodePro'])

# Encode categorical variables safely
from sklearn.preprocessing import LabelEncoder

for col in ['EdLevel', 'Employment', 'RemoteWork', 'JobSat']:
    clean_df.loc[:, col] = LabelEncoder().fit_transform(clean_df[col].astype(str))

# Features and target
features = ['EdLevel', 'Employment', 'RemoteWork', 'YearsCodePro']
target = 'ConvertedCompYearly'

X = clean_df[features]
y = clean_df[target]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 4))


# Hypothetical developer input
sample = pd.DataFrame({
    'EdLevel': [2],         # Example: Master's degree
    'Employment': [0],      # Example: Full-time
    'RemoteWork': [1],      # Example: Hybrid
    'YearsCodePro': [3.0]   # 3 years of coding experience
})

predicted_salary = model.predict(sample)
print("Predicted Salary (USD):", predicted_salary[0])
