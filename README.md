# 京东和淘宝评论数据处理与分析

本项目旨在处理和分析来自京东和淘宝两个平台的评论数据，主要针对女性护理产品的品牌和产品类别进行分类和统计，并进行基本的数据分析和可视化。

## 项目功能概述

该项目的代码通过以下步骤处理和分析京东和淘宝的评论数据：

1. **数据加载**: 从指定的文件夹中读取京东和淘宝的评论数据Excel文件。
2. **品牌分类**: 根据评论标题中的关键字，将评论归类到特定的品牌，如“苏菲”、“ABC”等。
3. **产品大类分类**: 根据关键字将评论归类到特定的产品类别，如“液体卫生巾”、“卫生棉条”等。
4. **数据增强**: 为评论添加来源平台、品牌名称、产品大类和评论字数等信息，增强数据的表达力。
5. **数据分析**: 通过 `data_analysis` 函数展示数据的概况，包括数据类型、唯一值数量和缺失值比例等。
6. **数据可视化**: 使用 `matplotlib` 和 `seaborn` 进行数据可视化，特别是针对中文字符的正确显示。

## 文件结构

- `dataset/京东评论/` - 存储所有京东评论数据的目录，包含多个Excel文件。
- `dataset/淘宝商品评论/` - 存储所有淘宝评论数据的目录，包含多个Excel文件。
- `results.json` - 包含每条评论的处理结果的JSON文件，用于进一步分析。
- `main.py` - 处理与分析京东和淘宝评论数据的主脚本。

## 使用方法

1. 将京东和淘宝评论数据分别放入各自的目录下，并确保 `results.json` 文件存在于项目根目录中。
2. 运行 `main.py` 脚本，脚本将自动处理数据、输出统计结果，并进行数据分析。

```bash
python main.py
`````
#输出示例

脚本运行后，终端将输出每个平台的数据中识别出的品牌名称和产品大类，以及它们的统计数量。例如：

##京东数据
```bash
文件名: jingdong_data1.xlsx
品牌统计:
{'苏菲': 120, 'ABC': 85, '护舒宝': 110}
产品大类统计:
{'液体卫生巾': 150, '卫生棉条': 50, '安心裤': 115}
`````
使用 data_analysis 函数可以展示数据的基本信息，如下所示：
```bash
| Column   | Data Type | Unique values | NaN |
|----------|-----------|---------------|-----|
| 页面标题 | object    | 10            | 0.0 |
| 评论内容 | object    | 500           | 0.02|
| ...      | ...       | ...           | ... |
`````

# 关键信息提取与情感分析

请参考`llm_out`,本项目利用大语言模型（GLM-4-9B）对电商平台上的商品评论进行情感分析与关键词提取，帮助用户快速了解评论的主要内容和情感倾向。
## 功能概述

- **情感分析**：根据商品评论的内容，提供从1星到5星的情感评分，1星代表负面评价，5星代表正面评价。
- **关键词提取**：从评论中提取与产品质量、使用体验、包装、价格等相关的关键词。

## 文件内容

- `call_qwen_api`：用于调用大语言模型API，对用户输入的商品评论进行分析，并返回关键词和情感评分。
- `api_retry`：提供API调用的重试机制，确保在调用失败时自动重试。
- `get_prompt`：定义了用于情感分析和关键词提取的prompt模板，适用于不同类别的产品评论。

该部分代码从每条评论中提取关键信息并进行情感分析，输出包括产品质量、使用体验、包装和价格的关键词，以及整体情感评分。

## 提取关键信息

从 results.json 文件中加载每条评论的处理结果，并提取以下信息：

产品质量关键词: 提取与产品质量相关的关键词。
使用体验关键词: 提取与使用体验相关的关键词。
包装关键词: 提取与包装相关的关键词。
价格关键词: 提取与价格相关的关键词。
情感评分: 提取评论的情感评分，范围通常为1-5星。如果评分缺失，则默认设置为3星。
提取的信息将与之前整合的评论数据合并，最终生成一个包含所有关键信息的数据集，并保存为 final_extracted.xlsx。

## 情感评分提取与可视化

提取情感评分后，使用柱状图展示各个情感评分的分布情况。

以下代码生成情感评分分布的柱状图，并显示每个评分对应的具体数量：
```bash
# 统计每个情感评分的数量
score_counts = df_extracted['情感评分'].value_counts().sort_index()

# 创建柱状图
plt.figure(figsize=(10, 10))
sns.barplot(x=score_counts.index, y=score_counts.values, palette="Blues_d")

# 显示每个柱子的具体数值
for index, value in enumerate(score_counts.values):
    plt.text(index, value + 200, str(value), ha='center', va='bottom', fontsize=12)

# 添加标题和标签
plt.title('情感评分分布', fontsize=16)
plt.xlabel('情感评分', fontsize=14)
plt.ylabel('数量', fontsize=14)

plt.show()
`````
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/c61d583d-2cf0-4734-af88-f6678bcb6f3d" alt="情感评分柱状图" style="width: 45%;" />
    <img src="https://github.com/user-attachments/assets/7d39b671-81be-4b57-a255-6d3dcf86f241" alt="数据分析结果" style="width: 45%;" />
</div>

# 分析数据

## 时间分布

接下来我们分别来关注女性用品评论量关于时间的分布，我们分别关注汇总，京东，淘宝三个维度，按照季度分别关注评论量。以汇总商品为例:

```bash
# 按月度统计评论数量
df = df_extracted.copy()
# 先将 '评价时间' 列转换为 datetime 类型
df['time'] = pd.to_datetime(df['评价时间'], errors='coerce')

# 转换成功后再执行后续操作，按季度统计
quarterly_counts = df['time'].dt.to_period('Q').value_counts().sort_index()

# 绘制评论的时间分布柱状图
plt.figure(figsize=(24, 12))
quarterly_counts.plot(kind='bar', color='skyblue')

# 设置图表标题和标签
plt.title('电商平台女性日用品商品汇总评论的时间分布（按季度）')
plt.xlabel('时间（季度）')
plt.ylabel('评论数量')

# 显示具体的数字
for index, value in enumerate(quarterly_counts):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=18)

plt.grid(True, axis='y')  # 仅显示Y轴的网格线
plt.show()
`````

<div align="center">
    <img src="https://github.com/user-attachments/assets/5e9178fc-c88f-4584-846b-e1036f5b4fd2" alt="评论时间分布柱状图" />
</div>

 那么根据这种类似的柱状图绘制，我们还可以绘制商品间的差异化分析图，平台差异化分析图，与品牌差异性分析图
 
商品差异化分析图（以卫生巾为例）：

<div align="center">
    <img src="https://github.com/user-attachments/assets/b46b98ef-2049-4b17-9762-47303c417ff1" alt="情感评分分布柱状图" />
</div>


平台差异化分析图（以京东平台卫生巾类为例）：

<div align="center">
    <img src="https://github.com/user-attachments/assets/d5898a2e-8366-45c3-9dd1-ae7fd47bee63" alt="按季度评论数量分布" />
</div>


品牌差异化分析图（以ABC品牌为例）：

<div align="center">
    <img src="https://github.com/user-attachments/assets/98fc7a0b-a1db-4232-8fcf-9dca440aff7f" alt="图表示例" />
</div>

# 评论字数情况

## 不同品牌在不同平台的评论字数分布分析

该部分代码旨在对不同品牌在京东和淘宝两个平台上的评论字数进行比较分析，并通过箱线图展示不同品牌和平台的评论字数分布情况。

### 功能

1. **数据筛选**:
   - 根据评论数据中的品牌大类筛选出卫生巾类别的数据。
   
2. **数据汇总**:
   - 对于每个品牌，获取在京东和淘宝平台上的评论字数数据，并按季度汇总。

3. **箱线图绘制**:
   - 为每个品牌绘制不同平台的评论字数分布箱线图。
   - 通过不同的颜色区分京东和淘宝平台。
   - 显示每个品牌的评论字数分布情况，便于比较不同品牌和平台的评论数量差异。

### 箱线图展示

箱线图将展示每个品牌在京东和淘宝平台上的评论字数分布，图示化信息包括：
- **品牌名称**: X轴显示不同品牌。
- **评论字数**: Y轴显示评论字数的分布情况。
- **平台区分**: 使用不同颜色表示京东（天蓝色）和淘宝（浅绿色）的数据。

#### 示例代码

```python
colors = {'京东': 'skyblue', '淘宝': 'lightgreen'}
# 筛选出卫生巾类别的数据
category_list = list(set(df['品牌大类']))

for category_name in category_list:
    df_weishengjin = df[df['品牌大类'] == category_name]

    # 获取所有品牌
    brands = df_weishengjin['品牌名称'].unique()

    # 获取所有平台
    platforms = ['京东', '淘宝']  # 根据数据的实际平台名调整

    # 初始化数据结构以存储每个品牌的评论字数数据
    brand_platform_data = {}

    for brand in brands:
        brand_platform_data[brand] = []
        for platform in platforms:
            # 获取该品牌在该平台上的评论字数，并按季度汇总
            word_counts_by_platform = df_weishengjin[(df_weishengjin['来源平台'] == platform) &
                                                     (df_weishengjin['品牌名称'] == brand)]['评论字数']
            brand_platform_data[brand].append(word_counts_by_platform)

    # 绘制箱线图
    plt.figure(figsize=(18, 12))

    # 用于控制不同品牌的位置
    positions = []

    for brand_index, brand in enumerate(brands):
        # 确定箱线图的横坐标位置
        positions.extend([brand_index * 2 + i * 0.5 for i in range(len(platforms))])

        # 绘制不同平台的箱线图
        for i, platform in enumerate(platforms):
            plt.boxplot(brand_platform_data[brand][i], positions=[brand_index * 2 + i * 0.5], widths=0.4,
                        patch_artist=True, boxprops=dict(facecolor=colors[platforms[i]]), medianprops=dict(color='red'))

    # 设置图表标题和标签
    plt.title(f'{category_name}不同品牌在不同平台的评论字数分布', fontsize=18)
    plt.xlabel('品牌名称', fontsize=18)
    plt.ylabel('评论字数', fontsize=18)
    plt.xticks([i * 2 + 0.25 for i in range(len(brands))], brands)  # 使品牌名与箱线图居中对齐

    # 添加图例
    handles = [plt.Line2D([0], [0], color=colors[platform], lw=4) for platform in platforms]
    plt.legend(handles, platforms, title="平台", fontsize=14, title_fontsize=16)

    # 显示网格
    plt.grid(True, axis='y')

    # 显示图表
    plt.show()
`````
以卫生巾在不同平台下的评论字数分布为例：

<p align="center">
  <img src="https://github.com/user-attachments/assets/edf89fac-0699-4566-a5bf-5760830ab587" alt="Image">
</p>

# 情感分析

我们之前用大语言模型——GLM-4-9B-Chat模型进行了情感评分，接下来我们将进一步分析具体商品间的用户满意度与评论情感。

该部分代码用于绘制不同品牌在不同情感评分下的评论数量柱状图。每个品牌的情感评分分布用不同的颜色表示，柱状图显示了每个品牌在各个评分下的评论数量。

图表说明
图表标题: 各品牌在不同情感评分下的评论数量分布
X轴: 品牌及其情感评分
Y轴: 评论数量
颜色映射:
1星: 蓝色
2星: 橙色
3星: 绿色
4星: 红色
5星: 紫色
每个品牌的柱状图展示了其在不同评分下的评论数量，图例中标明了不同情感评分的颜色。

以卫生巾为例：

```python
# 假设 df_weishengjin 已经包含所有数据
# 筛选出卫生巾类别的数据
# 获取所有品牌
category_list = list(set(df['品牌大类']))

for category_name in category_list:
    df_weishengjin = df[df['品牌大类'] == category_name]
    brands = df_weishengjin['品牌名称'].unique()

    # 定义情感评分的范围
    ratings = [1, 2, 3, 4, 5]

    # 定义颜色映射
    colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple'}

    # 为每个品牌绘制柱状图
    plt.figure(figsize=(18, 12))

    bar_width = 0.15  # 每个柱子的宽度
    space_between_brands = 0.5  # 品牌之间的间隔

    for brand_index, brand in enumerate(brands):
        # 初始化存储每个评分的评论数量
        rating_counts = []

        for rating in ratings:
            # 计算每个评分的评论数量
            count = df_weishengjin[(df_weishengjin['品牌名称'] == brand) & (df_weishengjin['情感评分'] == rating)].shape[0]
            rating_counts.append(count)

        # 计算每个柱子的X轴位置
        positions = np.arange(len(ratings)) * bar_width + brand_index * (bar_width * len(ratings) + space_between_brands)

        # 绘制每个情感评分的柱子
        for i, rating in enumerate(ratings):
            plt.bar(positions[i], rating_counts[i], color=colors[rating], width=bar_width, label=f'{rating}星' if brand_index == 0 else "")
            # 在每个柱子上显示具体的数值
            plt.text(positions[i], rating_counts[i] + 0.5, str(rating_counts[i]), ha='center', va='bottom', fontsize=16)

    # 设置图表标题和标签
    plt.title(f'{category_name}在不同情感评分下的评论数量分布(GLM-4-9B-Chat大模型打分)', fontsize=20)
    plt.xlabel('品牌及情感评分', fontsize=20)
    plt.ylabel('评论数量', fontsize=20)

    # 设置x轴的刻度标签，使品牌名与柱状图居中对齐
    brand_labels = []
    for brand in brands:
        for rating in ratings:
            brand_labels.append(f'{rating}星')

    # 生成x轴标签
    tick_positions = [i * (bar_width * len(ratings) + space_between_brands) + (bar_width * len(ratings) - bar_width) / 2 for i in range(len(brands))]
    plt.xticks(tick_positions, brands, rotation=0, ha='right', fontsize=20)

    # 添加图例
    plt.legend(title="评分", fontsize=14, title_fontsize=16)

    # 显示网格
    plt.grid(True, axis='y')

    # 显示图表
    plt.show()
`````

<p align="center">
  <img src="https://github.com/user-attachments/assets/090bc36a-d4b3-4c5b-bd01-af6ac899efcb" alt="Bar Chart">
</p>

接下来是情感评分的平均值：

```python
# 假设 df_weishengjin 已经包含所有数据
category_list = list(set(df['品牌大类']))

# 定义颜色映射
brand_colors = {
    '安尔乐': 'blue',
    '七度空间': 'orange',
    'ABC': 'green',
    '护舒宝': 'red',
    '丹碧丝': 'purple',
    '苏菲': 'pink',
    'ob': 'cyan'
}

# 创建一个包含四个子图的图表
fig, axes = plt.subplots(2, 2, figsize=(18, 20))  # 2x2网格布局

# 平铺模式下的索引
axes = axes.flatten()

for idx, category_name in enumerate(category_list):
    df_weishengjin = df[df['品牌大类'] == category_name]
    brands = df_weishengjin['品牌名称'].unique()

    # 为每个品牌计算情感评分的平均值
    average_ratings = []
    for brand in brands:
        avg_rating = df_weishengjin[df_weishengjin['品牌名称'] == brand]['情感评分'].mean()
        average_ratings.append(avg_rating)

    # 计算每个柱子的X轴位置
    positions = np.arange(len(brands))

    # 获取品牌对应的颜色
    colors = [brand_colors.get(brand, 'gray') for brand in brands]

    # 在子图上绘制柱状图
    axes[idx].bar(positions, average_ratings, color=colors, width=0.6)

    # 在每个柱子上显示具体的平均分数值
    for i, avg_rating in enumerate(average_ratings):
        axes[idx].text(positions[i], avg_rating + 0.05, f'{avg_rating:.2f}', ha='center', va='bottom', fontsize=20)

    # 设置子图的标题和标签
    axes[idx].set_title(f'{category_name}品牌评论在GLM-4-9B-Chat\n大模型打分所得平均分', fontsize=16)
    axes[idx].set_xlabel('品牌名称', fontsize=20)
    axes[idx].set_ylabel('平均评分', fontsize=20)

    # 设置X轴刻度标签为品牌名称
    axes[idx].set_xticks(positions)
    axes[idx].set_xticklabels(brands, rotation=0, ha='right', fontsize=20)

    # 显示网格
    axes[idx].grid(True, axis='y')

# 调整子图之间的间距
plt.tight_layout()

# 显示合并后的图表
plt.show()
`````
<p align="center">
  <img src="https://github.com/user-attachments/assets/8b40a451-4667-49db-ae5b-a422e821994f" alt="Box Plot">
</p>

# 词云图

## 关键词分词与词云生成

本部分代码的主要目的是对评论数据中的关键词进行分词处理，并生成高星和低星评论的词云图，以便更好地可视化评论的主要内容和情感倾向。

### 依赖库

确保你已经安装了以下 Python 库：
- `LAC`（用于中文分词）
- `wordcloud`（用于生成词云）
- `nltk`（用于停用词处理）

代码说明
分词处理

定义了 split_sentences 函数，通过标点符号切分句子。
使用 LAC 分词模型对切分后的句子进行分词，定义了 lac_cut_sentences 函数来处理这一任务。
对 DataFrame 中的每条评论的关键词进行分词处理，将 产品质量关键词、使用体验关键词、包装关键词 和 价格关键词 合并到一个新的列 关键词列表 中。
停用词处理

读取中文停用词文件 stopwords.txt，并将其中的词汇添加到停用词列表中。
对停用词列表去重，并在后续的分词处理时过滤掉这些停用词。
词云图生成

按 品牌大类 划分数据，对每个品牌类别生成高星和低星评论的词云图。
高星评论是指情感评分大于等于4的评论，低星评论是指情感评分小于等于2的评论。
使用 WordCloud 类生成词云图，并设置字体路径、图像大小及背景颜色。
绘制词云图

为每个品牌类别绘制两个词云图：一个表示高星评论的词云图，另一个表示低星评论的词云图。
使用 matplotlib 库绘制词云图，并将其展示出来。

```python
import re
from LAC import LAC
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 初始化LAC分词器
lac = LAC(mode='seg', use_cuda=True)

def split_sentences(text):
    sentences = re.split(r'[，。！？、,.!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def lac_cut_sentences(sentences):
    words = []
    for sentence in sentences:
        words.extend(lac.run(sentence))
    return words

df['关键词列表'] = df['产品质量关键词'].apply(lambda x: lac_cut_sentences(split_sentences(x)))
df['关键词列表'] = df['关键词列表'] + df['使用体验关键词'].apply(lambda x: lac_cut_sentences(split_sentences(x)))
df['关键词列表'] = df['关键词列表'] + df['包装关键词'].apply(lambda x: lac_cut_sentences(split_sentences(x)))
df['关键词列表'] = df['关键词列表'] + df['价格关键词'].apply(lambda x: lac_cut_sentences(split_sentences(x)))

from nltk.corpus import stopwords
chinese_stopwords = stopwords.words('chinese')
stopwords_path = './stopwords.txt'

with open(stopwords_path, 'r', encoding='utf-8') as file:
    words_from_file = file.read().split()

chinese_stopwords.extend(words_from_file)
stopwords = list(set(chinese_stopwords)) + ['无无', '无']

category_list = df['品牌大类'].unique()
font_path = 'C:/Windows/Fonts/simkai.ttf'

for category in category_list:
    df_category = df[df['品牌大类'] == category]
    high_rating_words = df_category[df_category['情感评分'] >= 4]['关键词列表'].sum()
    high_rating_words = [word for word in high_rating_words if len(word) >= 2]
    low_rating_words = df_category[df_category['情感评分'] <= 2]['关键词列表'].sum()
    low_rating_words = [word for word in low_rating_words if len(word) >= 2]

    high_rating_wordcloud = WordCloud(font_path=font_path, width=1800, height=1800, background_color='white').generate(' '.join(high_rating_words))
    low_rating_wordcloud = WordCloud(font_path=font_path, width=1800, height=1800, background_color='white').generate(' '.join(low_rating_words))

    plt.figure(figsize=(36, 18))
    plt.subplot(1, 2, 1)
    plt.imshow(high_rating_wordcloud, interpolation='bilinear')
    plt.title(f'{category} 好评词云图', fontsize=80)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(low_rating_wordcloud, interpolation='bilinear')
    plt.title(f'{category} 差评词云图', fontsize=80)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
`````
示例图片（以卫生巾为例）：

<p align="center">
  <img src="https://github.com/user-attachments/assets/ff30e32c-a33f-4544-be3b-cdefcaf4b54e" alt="image">
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/4fc12ce8-ecf8-498c-b951-b6cf36c7d5aa" alt="image">
</p>


