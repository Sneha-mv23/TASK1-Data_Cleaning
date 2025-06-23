import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

# 1.Import the dataset and explore basic info (nu ls, data types)
df = pd.read_csv("titanic_dataset.csv")  
print("Initial Shape:", df.shape)

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Null Values ---")
print(df.isnull().sum()) 

# 2.Handle missing values using mean/median/imputation.
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])


# 3.Convert categorical features into numerical using encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

#  4.Normalize/standardize the numerical features.
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 5.Visualize outliers using boxplots and remove them.
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("📦 Outlier Boxplot - Age & Fare")
plt.savefig("outliers_boxplot.png")
plt.close()

# Calculate IQR for Age
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age

# Calculate IQR for Fare
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

# Remove outliers for both Age and Fare
df = df[
    (df['Age'] >= (Q1_age - 1.5 * IQR_age)) & (df['Age'] <= (Q3_age + 1.5 * IQR_age)) &
    (df['Fare'] >= (Q1_fare - 1.5 * IQR_fare)) & (df['Fare'] <= (Q3_fare + 1.5 * IQR_fare))
]

print("\n Final Data Shape:", df.shape)
print(df.head())

df.to_csv("cleaned_data.csv", index=False)
print("\nCleaned data saved as 'cleaned_data.csv'")