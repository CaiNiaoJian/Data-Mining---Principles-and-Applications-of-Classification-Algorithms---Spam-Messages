import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 读取数据
def load_data(file_path):
    # 读取数据，使用制表符分隔，没有标题行
    df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    return df

# 数据预处理
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

# 训练和评估模型
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 初始化模型
    models = {
        '朴素贝叶斯': MultinomialNB(),
        '决策树': DecisionTreeClassifier(random_state=42),
        'K最近邻': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        # 生成分类报告
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report
        }
    
    return results

# 可视化结果
def plot_results(results):
    # 将中文模型名称映射为英文
    name_mapping = {
        '朴素贝叶斯': 'Naive Bayes',
        '决策树': 'Decision Tree',
        'K最近邻': 'K-NN'
    }
    
    names = [name_mapping[name] for name in results.keys()]
    accuracies = [results[name]['accuracy'] for name in results.keys()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies)
    plt.title('Accuracy Comparison of Different Classifiers')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    
    # 调整Y轴范围，使图表更加美观
    plt.ylim(0.90, 1.0)  # 将Y轴范围设置为0.90-1.0
    
    # 在柱状图上添加具体数值，调整文本位置
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center')  # 减小了文本与柱子的距离
    
    plt.savefig('classification_results.png')
    plt.close()

def main():
    # 加载数据
    print("正在加载数据...")
    df = load_data('SMSSpamCollection.txt')
    print(f"数据集大小: {len(df)} 条消息")
    print(f"垃圾短信比例: {(df['label'] == 'spam').mean():.2%}")
    
    # 预处理数据
    print("\n正在预处理数据...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)
    
    # 训练和评估模型
    print("\n正在训练和评估模型...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 输出结果
    print("\n分类结果：")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"准确率: {result['accuracy']:.4f}")
        print("详细分类报告:")
        print(result['report'])
    
    # 可视化结果
    print("\n正在生成可视化结果...")
    plot_results(results)
    print("结果已保存到 classification_results.png")

if __name__ == "__main__":
    main() 