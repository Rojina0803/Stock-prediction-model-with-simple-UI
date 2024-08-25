import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns

# Load dataset

dataset = pd.read_csv("data.csv")

# Drop null values
dataset.set_index("Date", inplace=True)
dataset.dropna(inplace=True)

# Set target variable and features
output_var = pd.DataFrame(dataset['Close'])
features = ['Open', 'High', 'Low', 'Volume']

# Scale features
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(dataset[features])
feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=dataset.index)

# Split data into training and testing sets
X = dataset.drop(['Close'], axis=1)
y = dataset['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
models = [
    RandomForestRegressor(n_estimators=500, random_state=42),
    DecisionTreeRegressor(random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10),
    KNeighborsRegressor(n_neighbors=7)
]

results = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test) * 100
    results.append((model.__class__.__name__, accuracy, mean_squared_error(y_test, y_pred)))

# Print results
for name, accuracy, mse in results:
    print(f"{name}: Accuracy={accuracy:.2f}%, MSE={mse:.2f}")

# Compare accuracy using seaborn
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel("TEST ACCURACY %")
plt.xlabel("MACHINE LEARNING ALGORITHMS")
sns.barplot(x=[model[0] for model in results], y=[model[1] for model in results])
plt.show()

#CREATING PICKLE FILE
import pickle
with open('stock_model.pkl','wb') as file:
    pickle.dump(model,file)