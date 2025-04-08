import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("webscraping_assignment5/cleaned_ebay_deals.csv")
print("Total rows loaded:", len(df))

df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])
print("Rows after cleaning:", len(df))


def convert_shipping(value):
    if value == "Free shipping":
        return 0
    try:
        
        return float(value.replace('$', '').replace(',', '')) 
    except ValueError:
        return np.nan 

df['shipping'] = df['shipping'].apply(convert_shipping)

df = df.dropna(subset=["shipping"])

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(df['discount_percentage'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Discount Percentage')
plt.xlabel('Discount Percentage')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

bins = [0, 10, 30, float('inf')]
labels = ['Low', 'Medium', 'High']
df['discount_bin'] = pd.cut(df['discount_percentage'], bins=bins, labels=labels, include_lowest=True)

print(df['discount_bin'].value_counts())

min_count = df['discount_bin'].value_counts().min()
df_balanced = (
    df.groupby('discount_bin', observed=True, group_keys=False)
    .apply(lambda x: x.sample(n=min_count, random_state=42), include_groups=False)
    .reset_index(drop=True)
)

X = df_balanced[['price', 'original_price', 'shipping']]
y = df_balanced['discount_percentage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("Model Evaluation Metrics:")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Actual vs Predicted Discount Percentages')
plt.xlabel('Actual Discount Percentage')
plt.ylabel('Predicted Discount Percentage')
plt.grid(True)
plt.tight_layout()
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, color='green', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Discount Percentage')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True)
plt.tight_layout()
plt.show()


df_incomplete = df.drop(columns=["discount_percentage"])

df_sample = df_incomplete.sample(n=20, random_state=42)[['title', 'price', 'original_price', 'shipping']]

X_new = df_sample[['price', 'original_price', 'shipping']]
df_sample['Predicted Discount (%)'] = model.predict(X_new)

print("\nPredicted Discounts for 20 Products:\n")
print(df_sample[['title', 'price', 'original_price', 'shipping', 'Predicted Discount (%)']].to_string(index=False))
