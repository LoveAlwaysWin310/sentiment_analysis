from transformers import pipeline
import json
import tqdm
import openai

classifier = pipeline('sentiment-analysis',model="nlp_structbert_emotion-classification_chinese-base")
openai.api_key = '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'

while True:
    print("--------------------------------------------")
    judge = str((input("输入你对这首歌的评论(按1进行情感分析）：")))
    if judge == '1':
        print("--------------------------------------------")
        print("进行评论的情感分析:")
        break
    else:
        # 使用聊天模型进行生成式回复
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 确保使用的是聊天模型
            messages=[
                {"role": "system", "content": "你是一个人类的朋友，你的任务是根据用户的评论生成人性化的回复。"},
                {"role": "user", "content": judge}
            ]
        )

        # 打印生成的文本
        print("对用户评论的回复："+response.choices[0].message['content'])
        print("评论的情感倾向是"+str(classifier(judge)[0]["label"])+"，概率为"+str(classifier(judge)[0]["score"]))

datas = json.load(open("data/comments.json", "r", encoding="utf-8"))

res = []
for data in tqdm.tqdm(datas):
    data["sentiment"] = classifier(data["text"])[0]["label"]
    res.append(data)

json.dump(res, open("data/comments_and_sentiment.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
