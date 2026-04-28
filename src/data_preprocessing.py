"""
data_preprocessing.py  —  FINAL VERSION
Run: python data_preprocessing.py
"""
import pandas as pd, numpy as np, joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Complete crop names mapping - 72 crops
CROP_NAMES = {
    0:"apple",1:"banana",2:"blackgram",3:"chickpea",4:"coconut",5:"coffee",
    6:"cotton",7:"grapes",8:"jute",9:"kidneybeans",10:"lemon",11:"lentil",
    12:"lettuce",13:"maize",14:"mango",15:"melon",16:"mothbeans",17:"mungbeans",
    18:"muskmelon",19:"nectarine",20:"orange",21:"papaya",22:"peach",23:"pear",
    24:"pigeonpeas",25:"pomegranate",26:"potato",27:"pumpkin",28:"radish",
    29:"rice",30:"soybean",31:"spinach",32:"strawberry",33:"sugarcane",
    34:"sweetpotato",35:"tomato",36:"watermelon",37:"wheat",
    # Additional crops (IDs 38-71) - Common Indian agricultural crops
    38:"groundnut",39:"mustard",40:"sunflower",41:"sesamum",42:"rapeseed",
    43:"barley",44:"oat",45:"tobacco",46:"rubber",47:"cashew",
    48:"arecanut",49:"tea",50:"blackpepper",51:"cardamom",52:"ginger",
    53:"turmeric",54:"coriander",55:"cumin",56:"fennel",57:"fenugreek",
    58:"garlic",59:"onion",60:"cauliflower",61:"cabbage",62:"carrot",
    63:"turnip",64:"beetroot",65:"parsley",66:"celery",67:"mint",
    68:"basil",69:"rosemary",70:"thyme",71:"drumstick"
}

SOIL_NAMES = {
    0:"alluvial",1:"black",2:"cinder",3:"clay",4:"coastal",5:"forest",
    6:"gravel",7:"laterite",8:"loam",9:"loamy",10:"marshy",11:"meadow",
    12:"mountainous",13:"peat",14:"peaty",15:"red",16:"red_loamy",17:"saline",
    18:"sandy",19:"sandy_loam",20:"silty",21:"silty_loam",22:"swampy",
    23:"terrace",24:"clay_loam",25:"silt",26:"chalky",27:"brown_forest",
    28:"podzolic",29:"tropical",30:"volcanic",31:"desert",32:"gypsic",
    33:"sodic",34:"permafrost",
}
SEASON_NAMES = {0:"kharif",1:"rabi",2:"zaid",3:"whole_year"}

FEATURE_COLS = ["N","P","K","temperature","humidity","ph","rainfall","soil","season","water"]
TARGET_COL   = "crop"
ENGINEERED_FEATURES = FEATURE_COLS + [
    "NPK_sum","N_ratio","P_ratio","K_ratio","heat_index","water_deficit"
]

def load_data(path):
    df = pd.read_csv(path)
    print(f"[INFO] Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df

def run_eda(df):
    print("\n" + "="*50)
    print("EDA REPORT")
    print("="*50)
    print(f"  Rows          : {len(df)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates    : {df.duplicated().sum()}")
    print(f"  Crop classes  : {df[TARGET_COL].nunique()}")
    print(f"  Min/Max samples per class: "
          f"{df[TARGET_COL].value_counts().min()} / "
          f"{df[TARGET_COL].value_counts().max()}")
    print("\nFeature stats:")
    print(df[FEATURE_COLS].describe().round(2).to_string())
    print("="*50)

def clean_data(df):
    before = len(df)
    df = df.drop_duplicates().dropna().copy()
    df["N"]           = df["N"].clip(0, 200)
    df["P"]           = df["P"].clip(0, 150)
    df["K"]           = df["K"].clip(0, 200)
    df["temperature"] = df["temperature"].clip(0, 50)
    df["humidity"]    = df["humidity"].clip(0, 100)
    df["ph"]          = df["ph"].clip(0, 14)
    df["rainfall"]    = df["rainfall"].clip(0, 300)
    df["water"]       = df["water"].clip(0, 3000)
    print(f"[INFO] Cleaned: removed {before-len(df)} bad rows, {len(df)} remain")
    return df

def engineer_features(df):
    df = df.copy()
    df["NPK_sum"]       = df["N"] + df["P"] + df["K"]
    df["N_ratio"]       = df["N"] / (df["NPK_sum"] + 1e-6)
    df["P_ratio"]       = df["P"] / (df["NPK_sum"] + 1e-6)
    df["K_ratio"]       = df["K"] / (df["NPK_sum"] + 1e-6)
    df["heat_index"]    = df["temperature"] * df["humidity"] / 100
    df["water_deficit"] = df["water"] - df["rainfall"]
    print("[INFO] Added 6 engineered features")
    return df

def prepare_splits(df, scaler_path=None, test_size=0.2, random_state=42):
    X = df[ENGINEERED_FEATURES].values
    y = df[TARGET_COL].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[INFO] Train: {len(X_train)}  Test: {len(X_test)}  Features: {X.shape[1]}")
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler saved -> {scaler_path}")
    return X_train, X_test, y_train, y_test, scaler, ENGINEERED_FEATURES

if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..")
    df   = load_data(os.path.join(BASE, "data/dataset.csv"))
    run_eda(df)
    df   = clean_data(df)
    df   = engineer_features(df)
    prepare_splits(df, scaler_path=os.path.join(BASE, "models/scaler.pkl"))
    print("\n[OK] Preprocessing done.\n")