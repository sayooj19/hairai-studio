import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv('Predict Hair Fall.csv')

# 2. Clean column names (removes trailing spaces like 'Nutritional Deficiencies ')
df.columns = df.columns.str.strip()

# 3. Define Features & Target
X = df.drop(columns=['Id', 'Hair Loss'])
y = df['Hair Loss']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 4. Preprocessing & ML Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42))
])

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_pipeline.fit(X_train, y_train)

# 6. Save Trained Model
joblib.dump(model_pipeline, 'hair_loss_kaggle_model.joblib')
print("Model successfully trained and saved as 'hair_loss_kaggle_model.joblib'")