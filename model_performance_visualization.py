import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_metrics_comparison():
    # 定义模型和指标数据
    models = ['Naive Bayes', 'Decision Tree', 'K-NN']
    metrics = {
        'Accuracy': [0.9857, 0.9731, 0.9256],
        'Precision': [0.9404, 0.9103, 1.0000],
        'Recall': [0.9530, 0.8859, 0.4430],
        'F1-Score': [0.9467, 0.8980, 0.6140]
    }
    
    # 创建柱状图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        plt.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    # 添加标签和标题
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 1.5, models)
    plt.legend(loc='lower left')
    plt.ylim(0.4, 1.05)  # 调整Y轴范围以更好地显示差异
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png')
    plt.close()

def plot_confusion_matrices():
    # 定义混淆矩阵数据
    confusion_matrices = {
        'Naive Bayes': np.array([[957, 9], [7, 142]]),
        'Decision Tree': np.array([[953, 13], [17, 132]]),
        'K-NN': np.array([[966, 0], [83, 66]])
    }
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, y=1.05)
    
    # 绘制每个模型的混淆矩阵
    for (title, cm), ax in zip(confusion_matrices.items(), axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Ham', 'Spam'])
        ax.set_yticklabels(['Ham', 'Spam'])
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png')
    plt.close()

def plot_error_analysis():
    # 错误分析数据
    models = ['Naive Bayes', 'Decision Tree', 'K-NN']
    false_positives = [9, 13, 0]  # 误判为垃圾的正常短信
    false_negatives = [7, 17, 83]  # 漏判的垃圾短信
    
    # 创建堆叠柱状图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x, false_positives, width, label='False Positives (Ham as Spam)',
            color='lightcoral')
    plt.bar(x, false_negatives, width, bottom=false_positives,
            label='False Negatives (Spam as Ham)', color='lightblue')
    
    # 添加标签和标题
    plt.xlabel('Models')
    plt.ylabel('Number of Errors')
    plt.title('Error Analysis Comparison')
    plt.xticks(x, models)
    plt.legend()
    
    # 在柱子上添加数值标签
    for i in range(len(models)):
        plt.text(i, false_positives[i]/2, str(false_positives[i]),
                ha='center', va='center')
        plt.text(i, false_positives[i] + false_negatives[i]/2,
                str(false_negatives[i]), ha='center', va='center')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.close()

def main():
    print("正在生成性能指标对比图...")
    plot_metrics_comparison()
    
    print("正在生成混淆矩阵对比图...")
    plot_confusion_matrices()
    
    print("正在生成错误分析图...")
    plot_error_analysis()
    
    print("\n可视化完成！生成了以下文件：")
    print("1. model_metrics_comparison.png - 模型性能指标对比")
    print("2. confusion_matrices_comparison.png - 混淆矩阵对比")
    print("3. error_analysis.png - 错误分析对比")

if __name__ == "__main__":
    main() 