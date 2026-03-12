import pandas as pd
import numpy as np

df = pd.read_csv("forest_fire_scaled.csv")
df['Classes'] = df['Classes'].map({'not fire':0, 'fire':1})

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = pd.to_numeric(df[column], errors='coerce')

df.to_csv("forest_fire_scaled.csv", index=False)

import numpy as np

raw_fwi = df['FWI']

conditions = [
    (raw_fwi < -0.54),
    (raw_fwi >= -0.54) & (raw_fwi < 0.85),
    (raw_fwi >= 0.85)
]

risk_levels = ['Low', 'Moderate', 'Severe']

df['Fire_Risk'] = np.select(conditions, risk_levels, default='Low')

print(df['Fire_Risk'].value_counts())

df['Fire_Risk_Label'] = df['Fire_Risk'].map({
    'Low': 0,
    'Moderate': 1,
    'Severe': 2
})

print(df[['FWI','Fire_Risk','Fire_Risk_Label']])

print(df['Fire_Risk'].value_counts())

df.to_csv("algerian_forest_fire_with_features.csv", index=False)

df['Temp_Humidity'] = df['Temperature'] * df['RH']
df['Temp_Humidity_Ratio'] = df['Temperature'] / df['RH']
df['Heat_Index'] = 0.5 * (df['Temperature'] + 61.0 + ((df['Temperature']-68.0)*1.2) + (df['RH']*0.094))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Fire_Risk_Encoded"] = le.fit_transform(df["Fire_Risk"])

print("Label Encoding mapping:")
for cls, val in zip(le.classes_, range(len(le.classes_))):
    print(f"{cls} -> {val}")

print(df[["Fire_Risk", "Fire_Risk_Encoded"]].drop_duplicates().sort_values("Fire_Risk_Encoded"))

from sklearn.preprocessing import StandardScaler, MinMaxScaler

cols_to_scale = [
    "Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC",
    "ISI", "BUI", "FWI", "Temp_Humidity", "Temp_Humidity_Ratio", "Heat_Index"
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(df_scaled[cols_to_scale].head())
print(df_scaled[cols_to_scale].mean().round(4))

mm_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[cols_to_scale] = mm_scaler.fit_transform(df[cols_to_scale])

print(df_minmax[cols_to_scale].head())
print(df_minmax[cols_to_scale].min())
print(df_minmax[cols_to_scale].max())

df_scaled.to_csv("forest_fire_scaled.csv", index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = df.drop(columns=[
    "Fire_Risk", "Fire_Risk_Label", "Fire_Risk_Encoded",
    "Classes","FWI","day","month","year"
])

y = df["Fire_Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Predictions:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

REAL_MEANS = {
    'Temperature':32.14,'RH':57.84,'Ws':16.85,'Rain':0.24,
    'FFMC':77.96,'DMC':15.16,'DC':80.11,'ISI':5.93,'BUI':17.18
}

REAL_STDS  = {
    'Temperature':5.85,'RH':18.08,'Ws':5.38,'Rain':1.04,
    'FFMC':17.23,'DMC':15.93,'DC':80.12,'ISI':7.15,'BUI':18.06
}

def scale(val, feat):
    return (val - REAL_MEANS[feat]) / REAL_STDS[feat]

def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))

            if min_val is not None and value < min_val:
                print("Must be >=", min_val)
                continue

            if max_val is not None and value > max_val:
                print("Must be <=", max_val)
                continue

            return value

        except ValueError:
            print("Invalid input")

def predict_fire_risk():

    print("FOREST FIRE RISK PREDICTION SYSTEM")

    Temperature = get_float_input("Temperature: ")
    RH = get_float_input("Humidity: ")
    Ws = get_float_input("Wind Speed: ")
    Rain = get_float_input("Rain: ")
    FFMC = get_float_input("FFMC: ")
    DMC = get_float_input("DMC: ")
    DC = get_float_input("DC: ")
    ISI = get_float_input("ISI: ")
    BUI = get_float_input("BUI: ")

    T_sc  = scale(Temperature, 'Temperature')
    RH_sc = scale(RH,'RH')
    Ws_sc = scale(Ws,'Ws')
    Rn_sc = scale(Rain,'Rain')
    FF_sc = scale(FFMC,'FFMC')
    DM_sc = scale(DMC,'DMC')
    DC_sc = scale(DC,'DC')
    IS_sc = scale(ISI,'ISI')
    BU_sc = scale(BUI,'BUI')

    Temp_Humidity = Temperature * RH
    Temp_Humidity_Ratio = Temperature / RH if RH != 0 else 0
    Heat_Index = 0.5 * (Temperature + 61 + ((Temperature-68)*1.2) + (RH*0.094))

    model_input = pd.DataFrame([{
        'Temperature':T_sc,
        'RH':RH_sc,
        'Ws':Ws_sc,
        'Rain':Rn_sc,
        'FFMC':FF_sc,
        'DMC':DM_sc,
        'DC':DC_sc,
        'ISI':IS_sc,
        'BUI':BU_sc,
        'Temp_Humidity':Temp_Humidity,
        'Temp_Humidity_Ratio':Temp_Humidity_Ratio,
        'Heat_Index':Heat_Index
    }])

    prediction = rf_model.predict(model_input)[0]

    label_map = {
        0:'Low',
        1:'Moderate',
        2:'Severe'
    }

    print("Predicted Fire Risk:", label_map[prediction])

predict_fire_risk()