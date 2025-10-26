import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')


def main():
    # Load dataset
    df = pd.read_csv('Tesla.csv')  # Replace with full path if needed
    print(df.head())
    print("Shape:", df.shape)
    print(df.describe())
    print(df.info())

    # Plot Close price
    plt.figure(figsize=(15, 5))
    plt.plot(df['Close'])
    plt.title('Tesla Close price.', fontsize=15)
    plt.ylabel('Price in dollars.')
    plt.savefig("close_price.png")  # Save plot
    plt.close()

    # Drop Adj Close if same as Close
    if (df['Close'] == df['Adj Close']).all():
        df = df.drop(['Adj Close'], axis=1)

    # Check for nulls
    print("Null values:\n", df.isnull().sum())

    # Plot distributions
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sb.histplot(df[col], kde=True)
    plt.tight_layout()
    plt.savefig("distributions.png")
    plt.close()

    # Plot boxplots
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sb.boxplot(x=df[col])
    plt.tight_layout()
    plt.savefig("boxplots.png")
    plt.close()

    # Feature engineering
    splitted = df['Date'].str.split('/', expand=True)
    df['day'] = splitted[1].astype(int)
    df['month'] = splitted[0].astype(int)
    df['year'] = splitted[2].astype(int)
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

    # Yearly mean plots
    data_grouped = df.drop('Date', axis=1).groupby('year').mean()
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
        plt.subplot(2, 2, i + 1)
        data_grouped[col].plot.bar(title=col)
    plt.tight_layout()
    plt.savefig("yearly_averages.png")
    plt.close()

    print("Quarterly mean:\n", df.drop('Date', axis=1).groupby('is_quarter_end').mean())

    # Target feature
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Pie chart
    plt.pie(df['target'].value_counts().values, 
            labels=[0, 1], autopct='%1.1f%%')
    plt.savefig("target_distribution.png")
    plt.close()

    # Heatmap of correlations
    plt.figure(figsize=(10, 10))
    sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
    plt.savefig("high_correlations.png")
    plt.close()

    # Prepare data
    features = df[['open-close', 'low-high', 'is_quarter_end']]
    target = df['target']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features, target, test_size=0.1, random_state=2022)
    print("Train/Validation Shapes:", X_train.shape, X_valid.shape)

    # Models
    models = [LogisticRegression(), 
              SVC(kernel='poly', probability=True), 
              XGBClassifier(use_label_encoder=False, eval_metric='logloss')]

    # Train and evaluate
    for model in models:
        model.fit(X_train, Y_train)
        print(f'{model.__class__.__name__} :')
        print('Training AUC :', metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
        print('Validation AUC :', metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
        print("Classification Report:")
        print(classification_report(Y_valid, model.predict(X_valid)))
        print()

        # Confusion Matrix
        disp = ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid)
        plt.title(f"Confusion Matrix - {model.__class__.__name__}")
        plt.savefig(f"conf_matrix_{model.__class__.__name__}.png")
        plt.close()

    # ROC Curve
    plt.figure(figsize=(10, 8))
    for model in models:
        fpr, tpr, _ = roc_curve(Y_valid, model.predict_proba(X_valid)[:, 1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid()
    plt.savefig("roc_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
