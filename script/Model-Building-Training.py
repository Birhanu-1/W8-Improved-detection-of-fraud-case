import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler
# Helper function for evaluation
def evaluate_model(y_true, y_pred, y_scores=None):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    if y_scores is not None:
        avg_precision = average_precision_score(y_true, y_scores)
        print(f"Average Precision (AUC-PR): {avg_precision:.4f}")
    
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1:.4f}")
    return avg_precision, f1
# Prepare and train models for a datase
def train_and_evaluate(X, y, dataset_name="Dataset"):
    print(f"\n=== {dataset_name} ===")
    # Train-test split with stratification to keep imbalance ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale numeric features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance with SMOTE (if you have it), else comment this and train on original
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_res, y_train_res)
    y_pred_logreg = logreg.predict(X_test_scaled)
    y_score_logreg = logreg.predict_proba(X_test_scaled)[:, 1]

    print("\nLogistic Regression Performance:")
    evaluate_model(y_test, y_pred_logreg, y_score_logreg)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test_scaled)
    y_score_rf = rf.predict_proba(X_test_scaled)[:, 1]

    print("\nRandom Forest Performance:")
    evaluate_model(y_test, y_pred_rf, y_score_rf)

    # Determine "best" model by Average Precision and F1-score
    ap_logreg, f1_logreg = evaluate_model(y_test, y_pred_logreg, y_score_logreg)
    ap_rf, f1_rf = evaluate_model(y_test, y_pred_rf, y_score_rf)

    if ap_rf > ap_logreg and f1_rf > f1_logreg:
        print("\n=> Random Forest is the best model based on AUC-PR and F1-score.")
    else:
        print("\n=> Logistic Regression is the best model based on AUC-PR and F1-score.")
# For Fraud_Data dataset
# -------- Task 1: Data Loading, Merging, Feature Engineering --------

def to_ip_int(val):
    if pd.isna(val):
        return np.nan
    try:
        return int(ipaddress.ip_address(val))
    except:
        try:
            return int(float(val))
        except:
            return np.nan

def task1_prepare_fraud_data(fraud_path, ip_path):
    fraud_df = pd.read_csv('../data/Fraud_Data.csv')
    ip_df = pd.read_csv('../data/IpAddress_to_Country.csv')

    fraud_df['ip_int'] = fraud_df['ip_address'].apply(to_ip_int)
    ip_df['lower'] = ip_df['lower_bound_ip_address'].apply(to_ip_int)
    ip_df['upper'] = ip_df['upper_bound_ip_address'].apply(to_ip_int)

    fraud_df = fraud_df.dropna(subset=['ip_int'])
    ip_df = ip_df.dropna(subset=['lower'])

    fraud_df = fraud_df.sort_values('ip_int').reset_index(drop=True)
    ip_df = ip_df.sort_values('lower').reset_index(drop=True)

    merged_df = pd.merge_asof(fraud_df, ip_df, left_on='ip_int', right_on='lower', direction='backward')

    merged_df['purchase_time'] = pd.to_datetime(merged_df['purchase_time'], errors='coerce')
    merged_df['signup_time'] = pd.to_datetime(merged_df['signup_time'], errors='coerce')

    merged_df = merged_df.dropna(subset=['purchase_time', 'signup_time'])

    merged_df['hour_of_day'] = merged_df['purchase_time'].dt.hour
    merged_df['day_of_week'] = merged_df['purchase_time'].dt.dayofweek
    merged_df['time_since_signup'] = (merged_df['purchase_time'] - merged_df['signup_time']).dt.total_seconds()

    user_freq = merged_df.groupby('user_id').size().rename('user_transaction_count')
    merged_df = merged_df.merge(user_freq, on='user_id')

    # Encode categorical columns if any
    cat_cols = merged_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        merged_df[col] = merged_df[col].astype('category').cat.codes

    return merged_df

def task1_prepare_creditcard_data(creditcard_path):
    credit_df = pd.read_csv("../data/creditcard.csv")
    return credit_df
# -------- Task 2: Model Building, Training & Evaluation --------

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Average Precision (AUC-PR): {average_precision_score(y_test, y_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

def run_models(X, y, dataset_name):
    print(f"\n===== {dataset_name} =====")
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    print("\nLogistic Regression Results:")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test)

    print("\nRandom Forest Results:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test)

# -------- Main --------

if __name__ == "__main__":
    fraud_data_path = '../data/Fraud_Data.csv'
    ip_data_path = '../data/IpAddress_to_Country.csv'
    creditcard_data_path = '../data/creditcard.csv'

    # Prepare datasets (Task 1)
    merged_df = task1_prepare_fraud_data(fraud_data_path, ip_data_path)
    credit_df = task1_prepare_creditcard_data(creditcard_data_path)

    # Prepare features and targets for Task 2
    y_fraud = merged_df['class']
    X_fraud = merged_df.drop(columns=['class', 'signup_time', 'purchase_time', 'ip_address', 'ip_int'])

    y_cc = credit_df['Class']
    X_cc = credit_df.drop(columns=['Class'])

    # Run models (Task 2)
    run_models(X_fraud, y_fraud, dataset_name="Fraud Detection Dataset")
    run_models(X_cc, y_cc, dataset_name="Credit Card Dataset")


