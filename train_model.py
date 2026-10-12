"""
Credit Card Fraud Detection - Model Training Pipeline
=====================================================
This script handles:
1. Data loading & Exploratory Data Analysis (EDA)
2. Preprocessing & Feature Engineering
3. Handling class imbalance with SMOTE
4. Training multiple models (Logistic Regression, Random Forest, XGBoost)
5. Model evaluation (Confusion Matrix, ROC Curve, Precision-Recall Curve)
6. Saving the best model for deployment
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# ============================================================
# Configuration
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set plot style
plt.style.use('dark_background')
sns.set_palette("bright")


def load_data():
    """Load the credit card fraud dataset."""
    print("=" * 60)
    print("📂 LOADING DATA")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Dataset not found at: {DATA_PATH}")
        print("\n📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"\n   Place 'creditcard.csv' in: {os.path.dirname(DATA_PATH)}")
        print("\n   Or run: kaggle datasets download -d mlg-ulb/creditcardfraud")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Columns: {df.shape[1]}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


def exploratory_data_analysis(df):
    """Perform and visualize EDA."""
    print("\n" + "=" * 60)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- Class Distribution ---
    fraud_count = df['Class'].value_counts()
    fraud_pct = df['Class'].value_counts(normalize=True) * 100
    print(f"\n📊 Class Distribution:")
    print(f"   Normal (0):     {fraud_count[0]:>8,} ({fraud_pct[0]:.3f}%)")
    print(f"   Fraudulent (1): {fraud_count[1]:>8,} ({fraud_pct[1]:.3f}%)")
    print(f"   Imbalance Ratio: 1:{fraud_count[0] // fraud_count[1]}")

    # --- Basic Statistics ---
    print(f"\n📈 Transaction Amount:")
    print(f"   Mean:   ${df['Amount'].mean():.2f}")
    print(f"   Median: ${df['Amount'].median():.2f}")
    print(f"   Max:    ${df['Amount'].max():.2f}")
    print(f"   Fraud Mean:   ${df[df['Class']==1]['Amount'].mean():.2f}")
    print(f"   Normal Mean:  ${df[df['Class']==0]['Amount'].mean():.2f}")

    print(f"\n⏱️  Time Span: {df['Time'].max() / 3600:.1f} hours")

    # --- Missing Values ---
    missing = df.isnull().sum().sum()
    print(f"\n🔲 Missing Values: {missing}")

    # ---- Chart 1: Class Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    colors = ['#00d4aa', '#ff4757']

    axes[0].bar(['Normal', 'Fraud'], fraud_count.values, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold', color='white')
    axes[0].set_ylabel('Count', color='white')
    for i, (v, p) in enumerate(zip(fraud_count.values, fraud_pct.values)):
        axes[0].text(i, v + 1000, f'{v:,}\n({p:.2f}%)', ha='center', fontsize=11, color='white', fontweight='bold')
    axes[0].set_facecolor('#161b22')

    axes[1].pie(fraud_count.values, labels=['Normal', 'Fraud'], colors=colors,
                autopct='%1.3f%%', startangle=90, textprops={'color': 'white', 'fontsize': 12},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
    axes[1].set_title('Fraud vs Normal Ratio', fontsize=14, fontweight='bold', color='white')
    axes[1].set_facecolor('#0d1117')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    # ---- Chart 2: Amount Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    axes[0].hist(df[df['Class'] == 0]['Amount'], bins=50, color='#00d4aa', alpha=0.7, label='Normal', edgecolor='none')
    axes[0].hist(df[df['Class'] == 1]['Amount'], bins=50, color='#ff4757', alpha=0.7, label='Fraud', edgecolor='none')
    axes[0].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold', color='white')
    axes[0].set_xlabel('Amount ($)', color='white')
    axes[0].set_ylabel('Frequency', color='white')
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(0, 2500)
    axes[0].set_facecolor('#161b22')

    axes[1].hist(df[df['Class'] == 0]['Time'] / 3600, bins=48, color='#00d4aa', alpha=0.7, label='Normal', edgecolor='none')
    axes[1].hist(df[df['Class'] == 1]['Time'] / 3600, bins=48, color='#ff4757', alpha=0.7, label='Fraud', edgecolor='none')
    axes[1].set_title('Transactions Over Time', fontsize=14, fontweight='bold', color='white')
    axes[1].set_xlabel('Time (hours)', color='white')
    axes[1].set_ylabel('Frequency', color='white')
    axes[1].legend(fontsize=11)
    axes[1].set_facecolor('#161b22')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'amount_time_distribution.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    # ---- Chart 3: Feature Correlation Heatmap ----
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')

    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_facecolor('#161b22')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    # ---- Chart 4: Top correlated features with Fraud ----
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')

    fraud_corr = df.corr()['Class'].drop('Class').sort_values()
    colors_bar = ['#ff4757' if x < 0 else '#00d4aa' for x in fraud_corr.values]
    fraud_corr.plot(kind='barh', color=colors_bar, ax=ax, edgecolor='none')
    ax.set_title('Feature Correlation with Fraud (Class)', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Correlation Coefficient', color='white')
    ax.axvline(x=0, color='white', linewidth=0.5, linestyle='--')
    ax.set_facecolor('#161b22')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'feature_correlation_fraud.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    print("📊 EDA charts saved to:", CHARTS_DIR)
    return df


def preprocess_data(df):
    """Preprocess data: scale features, split into train/test."""
    print("\n" + "=" * 60)
    print("⚙️  PREPROCESSING")
    print("=" * 60)

    # Use RobustScaler for Amount and Time (handles outliers better)
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()

    df['Scaled_Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Scaled_Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))

    # Drop original Amount and Time
    df = df.drop(['Amount', 'Time'], axis=1)

    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train/Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"✅ Train set: {X_train.shape[0]:,} samples")
    print(f"   Test set:  {X_test.shape[0]:,} samples")
    print(f"   Features:  {X_train.shape[1]}")
    print(f"   Train fraud rate: {y_train.mean()*100:.3f}%")
    print(f"   Test fraud rate:  {y_test.mean()*100:.3f}%")

    # Apply SMOTE to training data only
    print("\n🔄 Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"   Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"   After SMOTE:  {pd.Series(y_train_resampled).value_counts().to_dict()}")

    # Save scalers
    joblib.dump(scaler_amount, os.path.join(MODELS_DIR, 'scaler_amount.pkl'))
    joblib.dump(scaler_time, os.path.join(MODELS_DIR, 'scaler_time.pkl'))
    print("💾 Scalers saved")

    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    return X_train_resampled, X_test, y_train_resampled, y_test, feature_names


def train_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple models."""
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING")
    print("=" * 60)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0, solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=5, random_state=RANDOM_STATE,
            eval_metric='auc'
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"🔧 Training: {name}")
        print(f"{'─' * 40}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        trained_models[name] = model

        print(f"   Accuracy:          {accuracy:.4f}")
        print(f"   Precision:         {precision:.4f}")
        print(f"   Recall:            {recall:.4f}")
        print(f"   F1-Score:          {f1:.4f}")
        print(f"   ROC-AUC:           {roc_auc:.4f}")
        print(f"   Avg Precision:     {avg_precision:.4f}")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    return results, trained_models


def generate_evaluation_charts(results, y_test):
    """Generate evaluation comparison charts."""
    print("\n" + "=" * 60)
    print("📊 GENERATING EVALUATION CHARTS")
    print("=" * 60)

    model_names = list(results.keys())
    colors = {'Logistic Regression': '#3b82f6', 'Random Forest': '#10b981', 'XGBoost': '#f59e0b'}

    # ---- Chart 5: ROC Curves ----
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')

    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
        ax.plot(fpr, tpr, color=colors[name], linewidth=2.5,
                label=f"{name} (AUC = {results[name]['roc_auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
    ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=12, color='white')
    ax.set_ylabel('True Positive Rate', fontsize=12, color='white')
    ax.legend(fontsize=11, loc='lower right', facecolor='#161b22', edgecolor='#30363d')
    ax.set_facecolor('#161b22')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    # ---- Chart 6: Precision-Recall Curves ----
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')

    for name in model_names:
        prec, rec, _ = precision_recall_curve(y_test, results[name]['y_proba'])
        ax.plot(rec, prec, color=colors[name], linewidth=2.5,
                label=f"{name} (AP = {results[name]['avg_precision']:.4f})")

    ax.set_title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Recall', fontsize=12, color='white')
    ax.set_ylabel('Precision', fontsize=12, color='white')
    ax.legend(fontsize=11, loc='upper right', facecolor='#161b22', edgecolor='#30363d')
    ax.set_facecolor('#161b22')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'precision_recall_curves.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    # ---- Chart 7: Confusion Matrices ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0d1117')

    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_test, results[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
                    linewidths=2, linecolor='#0d1117',
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold', color='white', pad=10)
        axes[idx].set_xlabel('Predicted', fontsize=11, color='white')
        axes[idx].set_ylabel('Actual', fontsize=11, color='white')
        axes[idx].set_facecolor('#161b22')

    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    # ---- Chart 8: Metrics Comparison Bar Chart ----
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0d1117')

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[name],
                      edgecolor='none', alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_ylabel('Score', fontsize=12, color='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax.set_facecolor('#161b22')
    ax.grid(True, axis='y', alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'metrics_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    print("📊 Evaluation charts saved!")


def select_and_save_best_model(results, trained_models, feature_names):
    """Select the best model based on recall (priority for fraud detection) and save it."""
    print("\n" + "=" * 60)
    print("🏆 MODEL SELECTION")
    print("=" * 60)

    # For fraud detection, recall is most important (catching all frauds)
    # We use a weighted score: 0.4*recall + 0.3*f1 + 0.2*roc_auc + 0.1*precision
    best_name = None
    best_score = -1

    for name, metrics in results.items():
        weighted_score = (
            0.4 * metrics['recall'] +
            0.3 * metrics['f1_score'] +
            0.2 * metrics['roc_auc'] +
            0.1 * metrics['precision']
        )
        print(f"   {name}: weighted score = {weighted_score:.4f}")
        if weighted_score > best_score:
            best_score = weighted_score
            best_name = name

    print(f"\n🥇 Best Model: {best_name} (weighted score: {best_score:.4f})")

    # Save the best model
    best_model = trained_models[best_name]
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)

    # Save model info
    model_info = {
        'model_name': best_name,
        'weighted_score': float(best_score),
        'metrics': {
            'accuracy': float(results[best_name]['accuracy']),
            'precision': float(results[best_name]['precision']),
            'recall': float(results[best_name]['recall']),
            'f1_score': float(results[best_name]['f1_score']),
            'roc_auc': float(results[best_name]['roc_auc']),
            'avg_precision': float(results[best_name]['avg_precision']),
        },
        'feature_names': feature_names,
        'trained_at': datetime.now().isoformat(),
        'all_model_metrics': {}
    }

    for name, metrics in results.items():
        model_info['all_model_metrics'][name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'avg_precision': float(metrics['avg_precision']),
        }

    with open(os.path.join(MODELS_DIR, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    # Save all models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{safe_name}.pkl'))

    print(f"💾 Best model saved to: {model_path}")
    print(f"💾 All models saved to: {MODELS_DIR}")

    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0d1117')

        importance = pd.Series(best_model.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=True)

        colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))
        importance.plot(kind='barh', color=colors_bar, ax=ax, edgecolor='none')
        ax.set_title(f'Feature Importance ({best_name})', fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel('Importance Score', fontsize=12, color='white')
        ax.set_facecolor('#161b22')
        ax.grid(True, axis='x', alpha=0.15)

        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight',
                    facecolor='#0d1117', edgecolor='none')
        plt.close()
        print("📊 Feature importance chart saved!")

    return best_name, model_info


def main():
    """Main training pipeline."""
    print("\n" + "🔥" * 30)
    print("  CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("🔥" * 30 + "\n")

    # Step 1: Load Data
    df = load_data()

    # Step 2: EDA
    df = exploratory_data_analysis(df)

    # Step 3: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)[:4]
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

    # Step 4: Train Models
    results, trained_models = train_models(X_train, X_test, y_train, y_test)

    # Step 5: Evaluation Charts
    generate_evaluation_charts(results, y_test)

    # Step 6: Save Best Model
    best_name, model_info = select_and_save_best_model(results, trained_models, feature_names)

    # Final Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   ROC-AUC:   {model_info['metrics']['roc_auc']:.4f}")
    print(f"   Recall:    {model_info['metrics']['recall']:.4f}")
    print(f"   Precision: {model_info['metrics']['precision']:.4f}")
    print(f"   F1-Score:  {model_info['metrics']['f1_score']:.4f}")
    print(f"\n📁 Models saved in:  {MODELS_DIR}")
    print(f"📊 Charts saved in:  {CHARTS_DIR}")
    print(f"\n🚀 Run 'python app.py' to start the web application!")


if __name__ == '__main__':
    main()
