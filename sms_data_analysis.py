import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

def load_data(file_path):
    # 读取数据，使用制表符分隔，没有标题行
    df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    return df

def analyze_message_length(df):
    # 添加消息长度列
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    
    # 计算基本统计信息
    length_stats = df.groupby('label').agg({
        'message_length': ['mean', 'median', 'min', 'max'],
        'word_count': ['mean', 'median', 'min', 'max']
    })
    
    return length_stats

def plot_message_length_distribution(df):
    plt.figure(figsize=(15, 5))
    
    # 消息长度分布
    plt.subplot(1, 2, 1)
    sns.boxplot(x='label', y='message_length', data=df)
    plt.title('Message Length Distribution by Label')
    plt.xlabel('Label')
    plt.ylabel('Message Length (characters)')
    
    # 词数分布
    plt.subplot(1, 2, 2)
    sns.boxplot(x='label', y='word_count', data=df)
    plt.title('Word Count Distribution by Label')
    plt.xlabel('Label')
    plt.ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig('message_length_distribution.png')
    plt.close()

def analyze_common_words(df):
    # 分别统计ham和spam中的常见词
    def get_word_freq(messages):
        words = ' '.join(messages).lower()
        # 移除标点符号和数字
        words = re.sub(r'[^\w\s]', '', words)
        words = re.sub(r'\d+', '', words)
        return Counter(words.split())
    
    ham_words = get_word_freq(df[df['label'] == 'ham']['message'])
    spam_words = get_word_freq(df[df['label'] == 'spam']['message'])
    
    return ham_words.most_common(20), spam_words.most_common(20)

def plot_word_frequency(ham_words, spam_words):
    plt.figure(figsize=(15, 6))
    
    # Ham词频
    plt.subplot(1, 2, 1)
    words, counts = zip(*ham_words)
    plt.barh(range(len(counts)), counts)
    plt.yticks(range(len(counts)), words)
    plt.title('Most Common Words in Ham Messages')
    plt.xlabel('Frequency')
    
    # Spam词频
    plt.subplot(1, 2, 2)
    words, counts = zip(*spam_words)
    plt.barh(range(len(counts)), counts)
    plt.yticks(range(len(counts)), words)
    plt.title('Most Common Words in Spam Messages')
    plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('word_frequency.png')
    plt.close()

def plot_label_distribution(df):
    plt.figure(figsize=(8, 6))
    label_counts = df['label'].value_counts()
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'])
    plt.title('Distribution of SMS Labels')
    plt.savefig('label_distribution.png')
    plt.close()

def main():
    # 加载数据
    print("正在加载数据...")
    df = load_data('SMSSpamCollection.txt')
    
    # 基本统计信息
    print("\n数据集基本信息：")
    print(f"总消息数: {len(df)}")
    print("\n标签分布：")
    print(df['label'].value_counts())
    print("\n标签百分比：")
    print(df['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    # 分析消息长度
    print("\n消息长度统计：")
    length_stats = analyze_message_length(df)
    print("\n每类消息的长度统计：")
    print(length_stats)
    
    # 绘制消息长度分布图
    print("\n正在生成消息长度分布图...")
    plot_message_length_distribution(df)
    
    # 分析常见词
    print("\n正在分析常见词...")
    ham_words, spam_words = analyze_common_words(df)
    print("\nHam消息中最常见的词：")
    for word, count in ham_words[:10]:
        print(f"{word}: {count}")
    print("\nSpam消息中最常见的词：")
    for word, count in spam_words[:10]:
        print(f"{word}: {count}")
    
    # 绘制词频图
    print("\n正在生成词频分布图...")
    plot_word_frequency(ham_words, spam_words)
    
    # 绘制标签分布饼图
    print("\n正在生成标签分布饼图...")
    plot_label_distribution(df)
    
    print("\n分析完成！生成了以下可视化文件：")
    print("1. message_length_distribution.png - 消息长度和词数分布")
    print("2. word_frequency.png - 常见词频率分布")
    print("3. label_distribution.png - 标签分布饼图")

if __name__ == "__main__":
    main() 