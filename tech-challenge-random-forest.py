import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

data = pd.read_csv("data_en_us.csv")

print("=== Summary Statistics ===")
print(data.describe())
print("\n=== Correlations ===")
print(data.select_dtypes(include='number').corr())

# data visual exploration
sns.pairplot(data)
plt.show()

X = data.drop(columns=['charges'])
y = data['charges']

# splitting feature types
categorical_features = ['gender', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# pipeline for random forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# splitting training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# getting prediction metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# residuals plot
residuals = y_test - y_pred

print("\n=== Residuals Summary ===")
print(f"Mean Residual: {residuals.mean()}")
print(f"Standard Deviation of Residuals: {residuals.std()}")
print(f"Residuals Min: {residuals.min()}, Max: {residuals.max()}")

sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution Plot')
plt.axvline(0, color='red', linestyle='--')
plt.show()

# feature importance plot
feature_importances = model.named_steps['regressor'].feature_importances_
features = np.append(preprocessor.transformers_[0][2], model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n=== Feature Importances ===")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()
