import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
url = 'https://example.com/house_prices.csv'
df = pd.read_csv(url)

# Preprocessing
X = df[['feature1', 'feature2', 'feature3', 'feature4']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/house_price_model.pkl')
print('Model trained and saved successfully.')