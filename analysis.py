import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/used_cars.csv")

print(df.head())
print("\nDataset information:")
print(df.info())
print("\nStatistical summary:")
print(df.describe(include="all"))
print("\nMissing values:")
print(df.isnull().sum())

# Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["selling_price"].dropna(), kde=True)
plt.title("Distribution of Used Car Prices")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# Fuel type analysis
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="fuel")
plt.title("Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Year vs price
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="year", y="selling_price")
plt.title("Car Year vs Selling Price")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()

# Kilometers vs price
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="km_driven", y="selling_price")
plt.title("Kilometers Driven vs Selling Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()
