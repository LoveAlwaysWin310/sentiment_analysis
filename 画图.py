import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import jieba  # 导入jieba
from matplotlib.font_manager import FontProperties
from PIL import Image
from matplotlib import rcParams

font_path = "./微软雅黑.ttf"
font_prop = FontProperties(fname=font_path, size=14)

# 加载数据
data = json.load(open("data/comments_and_sentiment.json", "r", encoding="utf-8"))

# 转换为DataFrame
df = pd.DataFrame(data)

# 提取必要的信息
ip_counts = df['location'].apply(lambda x: x['location']).value_counts(normalize=True)
text_data = ' '.join(jieba.cut(' '.join(df['text'])))
sentiment_counts = df['sentiment'].value_counts()

sentiment_data = sentiment_counts.to_dict()
with open('data/情感分布图.json', 'w', encoding='utf-8') as f:
    # 使用json.dump将数据写入文件
    json.dump(sentiment_data, f, ensure_ascii=False, indent=4)

stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '为', '与', '你', '对', '说', '以', '而', '去', '能', '到', '之', '会'])
text_data_filtered = ' '.join([word for word in jieba.cut(' '.join(df['text'])) if word not in stopwords])
ip_counts_adjusted = ip_counts[ip_counts >= 0.03]
ip_counts_adjusted['其他'] = ip_counts[ip_counts < 0.03].sum()

data3 = ip_counts_adjusted.to_dict()
with open('data/IP属地图.json', 'w', encoding='utf-8') as f:
    # 使用json.dump将数据写入文件
    json.dump(data3, f, ensure_ascii=False, indent=4)

# 使用更美观的颜色方案绘制调整后的IP分布饼图
colors_adjusted = plt.cm.tab20c(np.linspace(0, 1, len(ip_counts_adjusted)))
plt.figure(figsize=(14, 7))
wedges, texts, autotexts = plt.pie(ip_counts_adjusted, autopct='%1.1f%%', startangle=140, colors=colors_adjusted, shadow=True, textprops={'fontsize': 12, 'fontproperties': font_prop}, wedgeprops={'edgecolor': 'black'})
plt.title('IP 分布图', fontproperties=font_prop, fontsize=15)
plt.legend(wedges, ip_counts_adjusted.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop=font_prop, title="IP")
plt.axis('equal')

# 调整标签位置
for text in texts:
    text.set_horizontalalignment('center')

# 添加数据标签
for i, autotext in enumerate(autotexts):
    autotext.set_color('black')  # 设置数据标签的颜色为黑色
    autotext.set_fontsize(12)  # 设置数据标签的字体大小
plt.savefig("picture/IP分布饼图")
plt.show()



# 生成并显示经过停用词过滤的词云图
mask = np.array(Image.open("logo.jpg"))
wordcloud_filtered = WordCloud(width=800, height=400, background_color='white', font_path=font_path, stopwords=stopwords,mask=mask).generate(text_data_filtered)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_filtered, interpolation='bilinear')
plt.axis('off')
plt.savefig("data/词云图")
plt.show()

# 绘制情感分析柱状图
plt.figure(figsize=(14, 7))
sentiment_counts.plot.bar()
plt.title('情感分析', fontproperties=font_prop)
plt.xlabel('情感', fontproperties=font_prop)
plt.ylabel('计数', fontproperties=font_prop)
plt.xticks(rotation=45, fontproperties=font_prop)
plt.savefig("picture/情感分析柱状图")
plt.show()