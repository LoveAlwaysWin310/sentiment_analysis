import os
import numpy
import pandas as pd
import jieba
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmgr
from snownlp import SnowNLP

# 正确显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


# 获取停用词
def get_stopwords_list():
    stopwords = [line.strip() for line in open(stopwords_file_name, encoding='UTF-8').readlines()]
    return stopwords


# 数据清洗，并生成词频
def comment_processing(comment_items):
    # 从临时保存文件获取数据
    stopwords = get_stopwords_list()  # 获取停用
    new_comment_items = []  # 清洗后的保存评论内容

    list1 = comment_items.values.tolist()
    print(list1)

    for row in comment_items.values.tolist():
        if row:
            # 进行分词，使用空格模式
            seg_arr = jieba.lcut(row[3])
            # print(seg_arr)
            for n_item in seg_arr:
                if n_item and len(n_item) > 1 and n_item != '\n' and n_item not in stopwords:
                    # 有词，且不在停用词内
                    new_comment_items.append(n_item)

    #print(new_comment_items)
    words_df = pd.DataFrame({'segment': new_comment_items})
    # stopwords = pandas.read_csv('stopword.txt',index_col=False,quoting=3,sep='，',names=['stopword'],encoding="utf-8")
    # words_df=words_df[~words_df.segment.isin(stopwords.stopword)]

    words_stat = words_df.groupby(by=['segment'])['segment'].agg(
        [("计数", numpy.size)]
    )

    words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)
    words_df.head()

    happy_arr = {}  # 正面的情感
    sadness_arr = {}  # 负面的情感
    total_score = 0  # 情感总分
    max_total_score = 0  # 满分

    total_score_arr = {'正面': 0, '负面': 0}
    for x in words_stat.values:
        # print(x)
        t_text, t_num = x
        t_score = SnowNLP(t_text).sentiments  # 获取情感分
        # print(t_score)
        if t_score >= 0.5:
            happy_arr[t_text] = [t_num, t_score]
            total_score_arr['正面'] += t_num
        else:
            sadness_arr[t_text] = [t_num, t_score]
            total_score_arr['负面'] += t_num

        total_score += t_score * t_num
        max_total_score += t_num

    t_idx = 0
    words_frequence = {}
    words_frequence_10 = {}
    for x in words_stat.values:
        # if t_idx<100:#只获取前100个
        words_frequence[x[0]] = x[1]
        if t_idx < 10:
            words_frequence_10[x[0]] = x[1]
        # else:
        #     break
        t_idx += 1
    # print(words_frequence)

    print('满分为：%s；总情感分：%s；分数比为：%.2f' % (max_total_score, total_score, (total_score / max_total_score * 100)))

    data2 = words_frequence_10
    with open('data/词频图.json', 'w', encoding='utf-8') as f:
        # 使用json.dump将数据写入文件
        json.dump(data2, f, ensure_ascii=False, indent=4)

    # 词频柱状图
    # word_frequenscy_file_name
    x = numpy.arange(len(words_frequence_10))
    y = words_frequence_10.values()
    plt.ylabel("数量", fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
    plt.xlabel("关键词", fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
    plt.title("评论词数统计", fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
    plt.bar(x, y)
    plt.xticks(x, words_frequence_10.keys(), fontproperties=fmgr.FontProperties(fname='./微软雅黑.ttf'))
    plt.savefig("picture/词频柱状图")
    plt.show()


if __name__ == "__main__":
    csv_path = 'music_comments.csv'
    stopwords_file_name = './stopwords.txt'  # 停用词

    data = pd.read_csv(csv_path,header=None)
    comment_processing(data)
    
