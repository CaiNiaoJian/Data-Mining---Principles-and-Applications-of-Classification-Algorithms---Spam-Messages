import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_data(file_path):
    # 读取数据，使用制表符分隔，没有标题行
    df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    return df

def preprocess_data(df):
    # 将标签转换为数值（ham=0, spam=1）
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 特征提取：将文本转换为词频向量
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer

def plot_confusion_matrix(cm, title, ax):
    # 创建混淆矩阵的��力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # 设置标签
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    # 设置刻度标签
    ax.set_xticklabels(['Ham', 'Spam'])
    ax.set_yticklabels(['Ham', 'Spam'])

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # 初始化模型
    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices for Different Classifiers', fontsize=16)
    
    # 训练模型并生成混淆矩阵
    results = {}
    for (name, model), ax in zip(models.items(), axes):
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(cm, name, ax)
        
        # 保存分类报告
        report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        results[name] = {
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    return results

def calculate_metrics(confusion_matrix):
    # 从混淆矩阵中提取值
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    # 计算各种指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }

def main():
    # 加载数据
    print("正在加载数据...")
    df = load_data('SMSSpamCollection.txt')
    
    # 预处理数据
    print("\n正在预处理数据...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)
    
    # 训练模型并生成混淆矩阵
    print("\n正在训练模型并生成混淆矩阵...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 输出详细的评估指标
    print("\n模型评估结果：")
    for name, result in results.items():
        print(f"\n{name}:")
        print("混淆矩阵��")
        print(result['confusion_matrix'])
        print("\n分类报告：")
        print(result['classification_report'])
        
        # 计算并显示额外的指标
        metrics = calculate_metrics(result['confusion_matrix'])
        print("\n额外指标：")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    print("\n分析完成！混淆矩阵可视化已保存到 confusion_matrices.png")

if __name__ == "__main__":
    main() 