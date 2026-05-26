import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

def prepare_datasets(df: pd.DataFrame, sample_ratio: float = 1.0, random_state: int = 42):
    """Cleans, engineers features, and encodes categorical columns."""
    df_prep = df.copy()

    df_prep['transactionmonth'] = pd.to_datetime(df_prep['transactionmonth'], errors='coerce')
    df_prep['vehicleintrodate'] = pd.to_datetime(df_prep['vehicleintrodate'], format='mixed', errors='coerce')

    df_prep['vehicle_age'] = df_prep['transactionmonth'].dt.year - df_prep['vehicleintrodate'].dt.year
    df_prep.loc[df_prep['vehicle_age'] < 0, 'vehicle_age'] = np.nan
    median_age = df_prep['vehicle_age'].median()
    df_prep['vehicle_age'] = df_prep['vehicle_age'].fillna(median_age if pd.notna(median_age) and median_age >= 0 else 5)

    df_prep['risk_density'] = df_prep['suminsured'] / (df_prep['vehicle_age'] + 1)
    df_prep['insured_per_age'] = df_prep['suminsured'] * df_prep['vehicle_age']

    df_prep['has_claim'] = np.where(df_prep['totalclaims'] > 0, 1, 0)

    feature_cols = [
        'vehicle_age', 'suminsured', 'risk_density', 'insured_per_age',
        'province', 'gender', 'maritalstatus', 'bodytype', 'make'
    ]
    feature_cols = [col for col in feature_cols if col in df_prep.columns]
    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")

    X_raw = df_prep[feature_cols].copy()

    num_cols = X_raw.select_dtypes(include='number').columns
    cat_cols = X_raw.select_dtypes(exclude='number').columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X_raw[num_cols] = imputer_num.fit_transform(X_raw[num_cols])

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_raw[cat_cols] = imputer_cat.fit_transform(X_raw[cat_cols])

    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    is_claim_mask = df_prep['totalclaims'] > 0
    X_reg = X_encoded[is_claim_mask].copy()
    y_reg = df_prep.loc[is_claim_mask, 'totalclaims']

    if sample_ratio < 1.0:
        logger.info(f"Downsampling negative class with sample_ratio={sample_ratio}...")
        pos_indices = df_prep[df_prep['has_claim'] == 1].index
        neg_indices = df_prep[df_prep['has_claim'] == 0].sample(
            frac=sample_ratio, random_state=random_state
        ).index
        sampled_indices = pos_indices.union(neg_indices)
        X_cls = X_encoded.loc[sampled_indices].copy()
        y_cls = df_prep.loc[sampled_indices, 'has_claim']
    else:
        X_cls = X_encoded.copy()
        y_cls = df_prep['has_claim']

    return X_reg, y_reg, X_cls, y_cls, X_encoded

def train_and_evaluate_regression(X_train, X_test, y_train, y_test, random_state: int = 42):
    """Trains and evaluates Linear Regression, Random Forest, and XGBoost Regressors."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric='rmse', random_state=random_state, n_jobs=-1
        )
    }

    results = {}
    fitted_models = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_log = np.log1p(y_train)
    max_log_val = y_train_log.max()

    for name, model in models.items():
        logger.info(f"Training Regressor: {name}...")
        model.fit(X_train_scaled, y_train_log)

        preds_log = model.predict(X_test_scaled)
        preds_log_clipped = np.clip(preds_log, 0, max_log_val)
        preds = np.expm1(preds_log_clipped)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4)
        }
        fitted_models[name] = model

    df_results = pd.DataFrame(results).T
    return df_results, fitted_models, scaler

def train_and_evaluate_classification(X_train, X_test, y_train, y_test, random_state: int = 42):
    """Trains and evaluates Logistic Regression, Random Forest, and XGBoost Classifiers."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced',
            random_state=random_state, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, scale_pos_weight=scale_weight,
            learning_rate=0.1, eval_metric='logloss',
            random_state=random_state, n_jobs=-1
        )
    }

    results = {}
    fitted_models = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        logger.info(f"Training Classifier: {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy  = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall    = recall_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)
        auc       = roc_auc_score(y_test, proba)

        results[name] = {
            'Accuracy':  round(accuracy,  4),
            'Precision': round(precision, 4),
            'Recall':    round(recall,    4),
            'F1-Score':  round(f1,        4),
            'ROC-AUC':   round(auc,       4)
        }
        fitted_models[name] = model

    df_results = pd.DataFrame(results).T
    return df_results, fitted_models, scaler
