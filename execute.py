import pandas as pd
import numpy as np
import argparse
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from scipy.stats import boxcox
from boruta import BorutaPy

### ---------- 清理服務欄位 ---------- ###
def clean_service_columns(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    internet_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_columns:
        df[col] = df[col].replace('No internet service', 'No')
    return df

### ---------- IQR 清除極端值 ---------- ###
def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

### ---------- Box-Cox ---------- ###
def boxcox_transform(df, column):
    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df[df[column] > 0]
    transformed_data, fitted_lambda = boxcox(df[column])
    df[column] = transformed_data
    print(f"\U0001f4e6 Box-Cox lambda for '{column}': {fitted_lambda:.4f}")
    return df

### ---------- SMOTENC ---------- ###
def apply_smotenc_corrected(df, target_col, numeric_cols):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_col] = y_resampled.reset_index(drop=True)
    print(f" SMOTENC 完成！原始樣本數: {len(df)}, 重抽後樣本數: {len(resampled_df)}")
    print(f" 新類別比例:\n{resampled_df[target_col].value_counts()}")
    return resampled_df

### ---------- 特徵選擇交集 ---------- ###
def select_features_with_intersection(df, target_col='Churn'):
    numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
    categorical_cols = [col for col in df.columns if col not in numeric_cols + [target_col]]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    rfe_model = LogisticRegression(max_iter=1000)
    rfe = RFE(rfe_model, n_features_to_select=int(X.shape[1] / 2))
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_].tolist()

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42)
    boruta.fit(X.values, y.values)
    boruta_features = X.columns[boruta.support_].tolist()

    intersection = set(rfe_features) & set(boruta_features)
    print(" RFE Features:", len(rfe_features))
    print(" Boruta Features:", len(boruta_features))
    print(" Final Intersection Features:", intersection)
    return intersection

### ---------- Dummy 名稱轉回原始特徵 ---------- ###
def extract_original_features_from_dummy(selected_features):
    original_feature_names = set()
    for feat in selected_features:
        if "_" in feat:
            original_feature_names.add(feat.split('_')[0])
        else:
            original_feature_names.add(feat)
    return list(original_feature_names)

### ---------- 隨機森林建模 ---------- ###
def train_random_forest_with_tuning(df, target_col='Churn', random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 5, 7, 9, 10],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=5,
                                scoring='f1', random_state=random_state, n_jobs=-1)
    search.fit(X, y)
    best_model = search.best_estimator_

    print(" 最佳超參數組合：", search.best_params_)

    y_pred = cross_val_predict(best_model, X, y, cv=5)
    report = classification_report(y, y_pred, output_dict=True)
    class_keys = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    positive_key = [k for k in class_keys if str(k) != '0'][0]

    print(" Classification Report:")
    print(f"Precision: {report[positive_key]['precision']:.4f}")
    print(f"Recall:    {report[positive_key]['recall']:.4f}")
    print(f"F1-Score:  {report[positive_key]['f1-score']:.4f}")

    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
    top5 = feature_importance.sort_values(ascending=False).head(5)

    print("\n Top 5 Most Important Features:")
    for feat, score in top5.items():
        print(f"{feat}: {score:.4f}")

    return best_model, top5

### ---------- 主流程 ---------- ###
def main(data_name):
    data = pd.read_csv(f"data/{data_name}.csv")
    data = clean_service_columns(data)
    data = remove_outliers_iqr(data, ['MonthlyCharges', 'TotalCharges', 'tenure'])
    data = boxcox_transform(data, 'TotalCharges')
    data = apply_smotenc_corrected(data, target_col='Churn', numeric_cols=['MonthlyCharges', 'TotalCharges', 'tenure'])

    selected_features = select_features_with_intersection(data)
    final_original_features = extract_original_features_from_dummy(selected_features)
    final_cols = final_original_features + ['Churn']
    final_data = data[final_cols]

    best_model, top5_features = train_random_forest_with_tuning(final_data, target_col='Churn')


### ---------- CLI 執行點 ---------- ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Dataset name (no extension). Must be in ./data folder.")
    args = parser.parse_args()
    main(args.data)

# python execute.py --data=Churn