import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmgr
import json

df = pd.read_csv('music_comments.csv')
df["time"] = df[df.columns[6]]
df['time'] = pd.to_datetime(df['time'])
df["date"] = df["time"].dt.date

comment_count = df.groupby('date').size()
comment_count_df = comment_count.reset_index()
comment_count_df.columns = ['date', 'num']  # 'date'是索引列的新名称，'num'是评论计数列的名称
data1 = comment_count.to_dict()
data1_dict = {date.strftime('%Y-%m-%d'): value for date, value in data1.items()}
with open('data/随时间分布图.json', 'w', encoding='utf-8') as f:
    # 使用json.dump将数据写入文件
    json.dump(data1_dict, f, ensure_ascii=False, indent=4)

# 指定要保存的CSV文件路径
file_path = 'comment_count.csv'

# 保存DataFrame到CSV文件
comment_count_df.to_csv(file_path, index=False)

# 计算指数加权移动平均
smoothed_comment_count = comment_count.ewm(span=10, adjust=False).mean()

# 设置图表风格为 seaborn-darkgrid
plt.style.use('seaborn-darkgrid')

# 绘制评论数量随时间的分布图
plt.figure(figsize=(10, 6))
plt.plot(smoothed_comment_count.index, smoothed_comment_count.values, label='Data', color='red')
plt.fill_between(smoothed_comment_count.index, smoothed_comment_count.values, color="red", alpha=0.1)  # 添加填充
plt.xlabel('日期',fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
plt.ylabel('评论数量',fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
plt.title('评论数量随时间变化趋势',fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.savefig("picture/评论随时间分布图")
plt.show()
