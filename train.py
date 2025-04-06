import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Load and prepare data
df = pd.read_csv("traffic.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Weekday'] = df['DateTime'].dt.weekday
df['Month'] = df['DateTime'].dt.month
df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)

# Categorize traffic
q1, q2 = df['Vehicles'].quantile([0.33, 0.66])
def categorize(v):
    if v <= q1:
        return 'Low'
    elif v <= q2:
        return 'Medium'
    else:
        return 'High'
df['Traffic_Level'] = df['Vehicles'].apply(categorize)

# Encode target
le = LabelEncoder()
df['Traffic_Level_Encoded'] = le.fit_transform(df['Traffic_Level'])

# Features and target
X = df[['Junction', 'Hour', 'Day', 'Weekday', 'Month', 'IsWeekend']]
y = df['Traffic_Level_Encoded']

# Train split and oversample
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train_res, y_train_res)

# Save model and label encoder
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model & encoder saved successfully!")
