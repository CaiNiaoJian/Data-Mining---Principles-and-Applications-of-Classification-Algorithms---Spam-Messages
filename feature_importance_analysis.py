import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

#这个新脚本专门用于分析决策树模型中特征的重要性，包含以下主要功能：
#特征提取和预处理：
#使用CountVectorizer提取文本特征
#限制为前50个最频繁的词，以保持可视化的清晰度
#特征重要性分析：
#使用决策树模型的feature_importances_属性获取特征重要性
#生成特征重要性排名
#可视化展示前20个最重要的特征
#特征分布分析：
#分析重要特征在垃圾短信和正常短信中的出现频率
#通过对比图展示特征的区分能力

def load_and_preprocess_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    
    # 将标签转换为数值
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 特征提取
    vectorizer = CountVectorizer(max_features=50)  # 限制为前50个最频繁的词
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    # 获取特征名称（词）
    feature_names = vectorizer.get_feature_names_out()
    
    return X, y, feature_names, vectorizer

def train_decision_tree(X, y):
    # 训练决策树模型
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X, y)
    return dt_classifier

def plot_feature_importance(model, feature_names):
    # 获取特征重要性
    importances = model.feature_importances_
    
    # 创建特征重要性的DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
    
    # 只显示重要性最高的20个特征
    top_20_features = feature_importance_df.tail(20)
    
    # 创建水平条形图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_20_features)), top_20_features['importance'])
    plt.yticks(range(len(top_20_features)), top_20_features['feature'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features (Words)')
    plt.title('Top 20 Most Important Features in Decision Tree Model')
    
    # 在每个条形上添加具体数值
    for i, v in enumerate(top_20_features['importance']):
        plt.text(v, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_importance_df

def analyze_spam_indicators(feature_importance_df, vectorizer, messages, labels):
    # 获取前10个最重要的特征
    top_features = feature_importance_df.tail(10)['feature'].values
    
    # 将稀疏矩阵转换为DataFrame
    feature_matrix = pd.DataFrame(
        vectorizer.transform(messages).toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    feature_matrix['label'] = labels
    
    # 分析每个重要特征在垃圾邮件和正常邮件中的出现频率
    plt.figure(figsize=(15, 6))
    
    for i, feature in enumerate(reversed(top_features)):
        # 计算在spam和ham中的出现比例
        spam_ratio = feature_matrix[feature_matrix['label'] == 1][feature].mean()
        ham_ratio = feature_matrix[feature_matrix['label'] == 0][feature].mean()
        
        # 绘制双条形图
        plt.subplot(1, 2, 1)
        plt.barh(i, spam_ratio, color='red', alpha=0.6)
        plt.barh(i, ham_ratio, color='blue', alpha=0.6)
    
    plt.yticks(range(len(top_features)), reversed(top_features))
    plt.xlabel('Average Occurrence')
    plt.title('Feature Occurrence in Spam vs Ham')
    plt.legend(['Spam', 'Ham'])
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('feature_occurrence_analysis.png')
    plt.close()

def main():
    print("正在加载和预处理数据...")
    X, y, feature_names, vectorizer = load_and_preprocess_data('SMSSpamCollection.txt')
    
    print("\n正在训练决策树模型...")
    dt_model = train_decision_tree(X, y)
    
    print("\n正在分析特征重要性...")
    feature_importance_df = plot_feature_importance(dt_model, feature_names)
    
    print("\n特征重要性排名（前10个）：")
    print(feature_importance_df.tail(10).to_string(index=False))
    
    print("\n正在分析特征在垃圾短信和正常短信中的分布...")
    # 读取原始数据用于分析
    df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['label', 'message'])
    analyze_spam_indicators(feature_importance_df, vectorizer, df['message'], 
                          df['label'].map({'ham': 0, 'spam': 1}))
    
    print("\n分析完成！生成了以下可视化文件：")
    print("1. feature_importance.png - 特征重要性排名")
    print("2. feature_occurrence_analysis.png - 特征在不同类别中的分布")

if __name__ == "__main__":
    main() 