import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from CSV file
df = pd.read_csv("heart.csv")

# Display data information
print(df.info())

# Check for missing data
print(df.isnull().sum())

#The dataset "heart.csv" does not contain any missing values, so the `dropna()` function is not needed.

# Map target values (0 represents no disease, 1 represents disease)
df["target"] = df.target.map({0: 0, 1: 1}) 
target_temp = df.target.value_counts()

print(target_temp)

countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])

# Visualize data
sns.set(style="whitegrid")

# Age distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Comparison plot of people with and without heart disease
plt.figure(figsize=(6, 6))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Diagnosis (0 = No, 1 = Yes)')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()

# Cholesterol levels by age plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='chol', hue='target', data=df)
plt.title('Cholesterol Levels by Age')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()
