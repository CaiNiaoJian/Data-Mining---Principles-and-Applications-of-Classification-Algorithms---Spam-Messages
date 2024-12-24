# SMS垃圾短信分类项目

这个项目使用机器学习方法对SMS短信进行分类，以识别垃圾短信（spam）和正常短信（ham）。项目实现了多个分类算法，并对其性能进行了详细的比较和分析。

## 环境要求

- Python 3.8+
- NumPy 1.24.3+
- Pandas 2.0.2+
- Scikit-learn 1.2.2+
- Matplotlib 3.7.1+
- Seaborn 0.13.2+

## 数据集介绍

SMS Spam Collection是一个公开的短信数据集，包含5,572条标记过的短信。

- **数据集大小**：5,572条消息
- **类别分布**：
  - 正常短信（Ham）：4,825条（86.59%）
  - 垃圾短信（Spam）：747条（13.41%）
- **数据格式**：每行包含两个字段，用制表符分隔
  - 标签（ham/spam）
  - 短信内容
- **数据特点**：
  - 类别不平衡（imbalanced dataset）
  - 文本长度变化大（2-910个字符）
  - 包含多种语言特征
  - 存在噪声（拼写错误、特殊字符等）

## 技术实现细节

### 数据预处理

1. **文本清洗**：
   - 移除标点符号和特殊字符
   - 转换为小写
   - 去除数字（可选）
   - 去除停用词（可选）

2. **特征提取**：
   - 使用CountVectorizer进行词频统计
   - 参数设置：
     ```python
     vectorizer = CountVectorizer(
         max_features=50,  # 限制特征数量
         stop_words=None,  # 不使用停用词
         token_pattern=r'\\b\\w+\\b'  # 词的正则表达式模式
     )
     ```

3. **数据集划分**：
   - 训练集：80%
   - 测试集：20%
   - 使用分层抽样保持类别比例
   - 随机种子：42（保证结果可复现）

### 算法实现细节

#### 1. 朴素贝叶斯（Naive Bayes）

```python
from sklearn.naive_bayes import MultinomialNB

# 模型初始化
nb_classifier = MultinomialNB()

# 模型训练
nb_classifier.fit(X_train, y_train)

# 概率预测
y_prob = nb_classifier.predict_proba(X_test)
```

**数学原理**：
- P(spam|text) = P(text|spam) × P(spam) / P(text)
- P(text|spam) = ∏ P(word_i|spam)
- 使用拉普拉斯平滑处理零概率问题
- 对数空间计算防止数值下溢

#### 2. 决策树（Decision Tree）

```python
from sklearn.tree import DecisionTreeClassifier

# 模型初始化
dt_classifier = DecisionTreeClassifier(
    criterion='gini',  # 使用基尼系数
    max_depth=None,    # 不限制树深度
    min_samples_split=2,  # 分裂节点所需的最小样本数
    random_state=42
)
```

**关键参数**：
- criterion：'gini'或'entropy'
- max_depth：控制树的深度
- min_samples_split：内部节点再划分所需最小样本数
- min_samples_leaf：叶子节点最少样本数

#### 3. K最近邻（K-NN）

```python
from sklearn.neighbors import KNeighborsClassifier

# 模型初始化
knn_classifier = KNeighborsClassifier(
    n_neighbors=5,     # 邻居数量
    weights='uniform', # 权重类型
    metric='minkowski' # 距离度量
)
```

**实现细节**：
- 使用KD树或球树加速近邻搜索
- 距离计算：闵可夫斯基距离（p=2时为欧氏距离）
- 时间复杂度：预测时O(d * log n)，d为特征维度

### 评估指标计算

1. **混淆矩阵**：
```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```

2. **性能指标**：
```python
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
```

3. **ROC曲线**：
```python
from sklearn.metrics import roc_curve, auc

# 获取预测概率
y_score = model.predict_proba(X_test)[:, 1]

# 计算ROC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
```

### 可视化实现

1. **特征重要性**：
```python
plt.figure(figsize=(12, 8))
importances = dt_classifier.feature_importances_
plt.barh(range(len(importances)), importances)
```

2. **ROC曲线**：
```python
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')  # 随机分类器基准线
```

3. **阈值分析**：
```python
plt.figure(figsize=(15, 5))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
```

## 性能优化技巧

1. **特征选择优化**：
   - 使用卡方检验选择最相关特征
   - 移除低频词（出现次数<5）
   - 考虑词组特征（n-grams）

2. **模型调优**：
   - 使用网格搜索优化超参数
   - 使用交叉验证评估模型稳定性
   - 对不平衡数据使用类别权重

3. **内存优化**：
   - 使用稀疏矩阵存储特征
   - 增量学习处理大数据集
   - 特征哈希减少维度

## 项目结构

项目文件组织如下：

1. 数据文件
   ├── SMSSpamCollection.txt    # 原始数据集
   └── datareadme.md           # 数据集说明文档

2. 依赖管理
   └── requirements.txt        # 项目依赖（numpy, pandas, scikit-learn, matplotlib, seaborn等）

3. 核心代码文件
   ├── sms_spam_classifier.py   # 主要的分类器实现
   ├── sms_data_analysis.py     # 数据分析脚本
   ├── feature_importance_analysis.py  # 特征重要性分析
   ├── confusion_matrix_analysis.py    # 混淆矩阵分析
   ├── model_performance_visualization.py  # 模型性能可视化
   └── roc_analysis.py          # ROC曲线分析

4. 可视化结果
   ├── word_frequency.png       # 词频分析结果
   ├── message_length_distribution.png  # 消息长度分布
   ├── label_distribution.png   # 标签分布
   ├── feature_importance.png   # 特征重要性
   ├── confusion_matrices.png   # 混淆矩阵
   ├── roc_curves.png          # ROC曲线
   └── precision_recall_threshold.png   # 精确率-召回率阈值曲线

5. 文档
   ├── README.md               # 项目说明文档
   └── result.md              # 结果分析文档

### 文件说明

#### 数据文件
- `SMSSpamCollection.txt`：包含5,572条标记过的短信数据
- `datareadme.md`：详细说明数据集的来源、格式和特点

#### 核心代码文件
- `sms_spam_classifier.py`：实现了主要的分类算法（朴素贝叶斯、决策树、K-NN）
- `sms_data_analysis.py`：进行数据探索性分析，包括文本长度、词频等统计
- `feature_importance_analysis.py`：分析和可视化特征重要性
- `confusion_matrix_analysis.py`：生成和分析混淆矩阵
- `model_performance_visualization.py`：创建模型性能的可视化图表
- `roc_analysis.py`：计算和绘制ROC曲线

#### 可视化结果
- `word_frequency.png`：展示垃圾短信和正常短信中最常见的词语
- `message_length_distribution.png`：显示不同类别短信的长度分布
- `label_distribution.png`：展示数据集中的类别分布
- `feature_importance.png`：展示不同特征对分类的重要性
- `confusion_matrices.png`：展示各个模型的混淆矩阵
- `roc_curves.png`：展示各个模型的ROC曲线
- `precision_recall_threshold.png`：展示不同阈值下的精确率和召回率

#### 文档文件
- `README.md`：项目的主要文档，包含项目说明、环境配置、实现细节等
- `result.md`：详细的实验结果分析和模型性能比较

## 实现的算法

### 1. 朴素贝叶斯（Naive Bayes）

#### 原理
- 基于贝叶斯定理，假设特征之间相互独立
- P(类别|文本) ∝ P(文本|类别) × P(类别)
- 使用词频作为特征，计算每个词在垃圾/正常短信中的条件概率

#### 优点
- 训练速度快
- 对小数据集效果好
- 对缺失数据不敏感
- 适合文本分类任务

#### 缺点
- 特征独立性假设在实际中往往不成立
- 对特征之间的相关性建模能力差

#### 性能表现
- 准确率：98.57%
- AUC值：0.9807
- 在本项目中表现最好

### 2. 决策树（Decision Tree）

#### 原理
- 通过递归划分特征空间构建树结构
- 每个节点代表一个特征的决策规则
- 使用信息增益或基尼系数选择最优划分特征

#### 优点
- 易于理解和解释
- 可以处理数值和分类特征
- 能自动进行特征选择
- 对异常值不敏感

#### 缺点
- 容易过拟合
- 可能产生过于复杂的树结构
- 对数据微小变化敏感

#### 性能表现
- 准确率：97.31%
- AUC值：0.9362
- 总体表现良好，但略逊于朴素贝叶斯

### 3. K最近邻（K-NN）

#### 原理
- 基于实例的学习方法
- 计算测试样本与训练样本的距离
- 取K个最近邻的多数类别作为预测结果

#### 优点
- 简单直观
- 无需训练过程
- 对异常样本不敏感
- 适合多分类问题

#### 缺点
- 计算复杂度高
- 对特征尺度敏感
- 需要大量内存存储训练数据
- 预测速度较慢

#### 性能表现
- 准确率：92.56%
- AUC值：0.8383
- 表现最保守，假阳性率低但漏报率高

## 性能评估指标

项目使用多个指标评估模型性能：

1. **准确率（Accuracy）**：正确分类的样本比例
2. **精确率（Precision）**：预测为垃圾短信中真正的垃圾短信比例
3. **召回率（Recall）**：实际垃圾短信中被正确识别的比例
4. **F1分数**：精确率和召回率的调和平均
5. **ROC曲线和AUC值**：评估模型在不同阈值下的表现
6. **混淆矩阵**：详细展示分类结果的分布

## 主要发现

1. **算法性能比较**：
   - 朴素贝叶斯性能最好，平衡性最佳
   - 决策树表现次之，但也达到了很好的效果
   - K-NN最保守，误报率低但漏报率高

2. **特征分析**：
   - "call"、"free"等词是识别垃圾短信的关键特征
   - 垃圾短信通常包含更多促销和行动号召用语
   - 正常短信用词更加个人化和多样化

3. **阈值分析**：
   - 通过调整决策阈值可以在精确率和召回率之间取得平衡
   - 不同应用场景可以选择不同的操作点

## 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行分类器：
```bash
python sms_spam_classifier.py
```

3. 查看分析结果：
```bash
python sms_data_analysis.py
python confusion_matrix_analysis.py
python feature_importance_analysis.py
python roc_analysis.py
```

## 未来改进方向

1. 特征工程优化：
   - 使用TF-IDF而不是简单的词频
   - 添加词组特征
   - 考虑词性标注特征

2. 模型优化：
   - 使用集成学习方法
   - 尝试深度学习模型
   - 进行超参数优化

3. 评估方法扩展：
   - 添加交叉验证
   - 考虑计算置信区间
   - 分析模型的稳定性 