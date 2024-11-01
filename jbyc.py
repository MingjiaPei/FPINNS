import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA

# 加载数据集
prs_data = pd.read_csv('Only_PGS.csv')
labels_data = pd.read_csv('all_labels.csv')


# 数据预处理
def preprocess_data(prs, labels):

    prs = prs.drop_duplicates()
    labels = labels.drop_duplicates()

    prs = prs.apply(pd.to_numeric, errors='coerce')
    labels = labels.apply(pd.to_numeric, errors='coerce')

    prs_na_eids = prs.loc[prs.isna().any(axis=1), 'eid']
    labels_na_eids = labels.loc[labels.isna().any(axis=1), 'eid']
    na_eids = pd.concat([prs_na_eids, labels_na_eids]).drop_duplicates()

    prs = prs[~prs['eid'].isin(na_eids)]
    labels = labels[~labels['eid'].isin(na_eids)]

    common_eids = prs['eid'].isin(labels['eid'])
    prs = prs[common_eids]
    labels = labels[labels['eid'].isin(prs['eid'])]
    prs_labels_merged = prs.merge(labels, on='eid', how='inner')

    X = prs_labels_merged.iloc[:, 1:81].values
    Y = prs_labels_merged.iloc[:, 81:].values

    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X)

    X_resampled, Y_resampled = resample(X_pca, Y, n_samples=len(X_pca), random_state=42)

    return X_resampled, Y_resampled, X_pca, prs_labels_merged['eid']


# 数据准备
X, Y, X_all, eids = preprocess_data(prs_data, labels_data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用 MultiOutputClassifier 来适应多标签任务
multi_rf = MultiOutputClassifier(rf, n_jobs=-1)
multi_rf.fit(X_train, Y_train)

# 预测概率
Y_pred_proba = multi_rf.predict_proba(X_test)

# 处理预测概率，确保没有越界错误
Y_pred = []
for proba in Y_pred_proba:
    if proba.shape[1] == 2:
        Y_pred.append(proba[:, 1])
    else:
        Y_pred.append(np.zeros(proba.shape[0]))

Y_pred = np.array(Y_pred).T

# 评估模型
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    auc_list = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            auc_list.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = np.mean(auc_list) if auc_list else float('nan')
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int), average='macro', zero_division=0)
    precision = precision_score(y_true, (y_pred > 0.5).astype(int), average='macro', zero_division=0)
    recall = recall_score(y_true, (y_pred > 0.5).astype(int), average='macro', zero_division=0)
    return {
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


metrics = evaluate_model(Y_test, Y_pred)


# 绘制评价指标
def plot_metrics(metrics):
    metric_names = list(metrics.keys())
    metrics_data = [metrics[metric] for metric in metric_names]

    plt.figure(figsize=(10, 7))
    plt.bar(metric_names, metrics_data, color='skyblue')
    plt.title("Evaluation Metrics for Random Forest Model")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


plot_metrics(metrics)

# 对所有样本进行预测
Y_pred_all = multi_rf.predict_proba(X_all)
Y_pred_X = []
for proba in Y_pred_all:
    if proba.shape[1] == 2:
        Y_pred_X.append(proba[:, 1])
    else:
        Y_pred_X.append(np.zeros(proba.shape[0]))

Y_pred_X = np.array(Y_pred_X).T

# 将预测结果保存到CSV文件
predictions_df = pd.DataFrame(Y_pred_X, columns=[f'Disease_{i + 1}_risk' for i in range(Y.shape[1])])
predictions_df.insert(0, 'eid', eids.values)
predictions_df.to_csv('disease_risk_predictions_rf.csv', index=False)




