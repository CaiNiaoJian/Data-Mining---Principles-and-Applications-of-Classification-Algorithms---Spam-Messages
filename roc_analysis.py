import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

#实现了以下功能：
#ROC曲线分析：
# 为每个模型计算并绘制ROC曲线
# 计算AUC值
# 添加随机分类器的基准线作为参考
#阈值分析：
# 分析不同决策阈值对精确率和召回率的影响
# 为每个模型生成阈值-精确率-召回率曲线
# 帮助选择最优决策阈值

def load_and_preprocess_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    
    # 将标签转换为数值
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 特征提取
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    # 初始化模型
    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # 训练模型
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    # 存储每个模型的AUC值
    auc_scores = {}
    
    # 为每个模型计算ROC曲线
    for name, model in models.items():
        # 获取预测概率
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 绘制随机分类器的基准线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()
    
    return auc_scores

def plot_precision_recall_threshold(models, X_test, y_test):
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items(), 1):
        # 获取预测概率
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        
        # 计算不同阈值下的精确率和召回率
        precisions = []
        recalls = []
        thresholds = np.linspace(0, 1, 100)
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 绘制阈值-精确率-召回率曲线
        plt.subplot(1, 3, i)
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{name}\nPrecision-Recall vs Threshold')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('precision_recall_threshold.png')
    plt.close()

def main():
    print("正在加载和预处理数据...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('SMSSpamCollection.txt')
    
    print("\n正在训练模型...")
    models = train_models(X_train, y_train)
    
    print("\n正在计算和绘制ROC曲线...")
    auc_scores = plot_roc_curves(models, X_test, y_test)
    
    print("\nAUC scores:")
    for name, score in auc_scores.items():
        print(f"{name}: {score:.4f}")
    
    print("\n正在分析不同阈值下的精确率和召回率...")
    plot_precision_recall_threshold(models, X_test, y_test)
    
    print("\n分析完成！生成了以下可视化文件：")
    print("1. roc_curves.png - ROC曲线对比")
    print("2. precision_recall_threshold.png - 阈值-精确率-召回率分析")

if __name__ == "__main__":
    main() 