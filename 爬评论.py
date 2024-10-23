import requests
# 实现AES加密需要的三个模块
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from base64 import b64encode
import json
import time
import pandas as pd

# py实现AES-CBC加密
def encrypt_aes(text, key, iv):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    padded_text = pad(text.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_text)
    return b64encode(ciphertext).decode('utf-8')

# 仿照function b(a, b)构造加密函数
def b(a,b):
    c=b
    d="0102030405060708"
    e=a
    f=encrypt_aes(e, c, d)
    return f

# 评论数据(i9b)
d={
    'csrf_token':'a377fd1409c2d967e66527ddf3ce2c02',#可以为空值
    'cursor': '-1',
    'offset': '0',
    'orderType': '1',
    'pageNo': '2',
    'pageSize': '100000',#评论数
    'rid': 'R_SO_4_150520',#歌曲编号
    'threadId': 'R_SO_4_150520'#歌曲编号
}

# 16位随机字符串
i="BdQMOhNkLlEP6jc7"
# bsu6o(["流泪", "强"])
e="010001"
# bsu6o(Xo0x.md)
f="00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7"
# bsu6o(["爱心", "女孩", "惊恐", "大笑"])
g="0CoJUm6Qyw8W8jud"

# 将i9b转化为json格式
d_json=json.dumps(d)

# 调用两次b()函数得出encText
encText=b(d_json,g)
encText=b(encText,i)

# 随机字符串获得encSecKey
encSecKey="1cac8643f7b59dbd626afa11238b1a90fab1e08bc8dabeec8b649e8a121b63fc45c2bc3427c6a9c6e6993624ec2987a2547c294e73913142444ddeec052b6ec2f9a4bebf57784d250e08749f371d94b635159a1c6ebfda81ee40600f2a22a5c1e7f0903884e4b466024a8905f0074a9432fd79c24ccf6aff73ea36fd68153031"

# 请求头
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
}

# 评论数据的请求地址
url='https://music.163.com/weapi/comment/resource/comments/get?csrf_token='

# 将encText和encSecKey打包起来
data={
'params':encText,
'encSecKey':encSecKey
}

# 发送post请求并携带encText和encSecKey得到评论的json格式
respond=requests.post(url, headers=headers,data=data).json()
items = respond["data"]['comments']

# 打印
for item in items:
    # 用户名
    user_name = item['user']['nickname'].replace(',', '，')
    # 用户ID
    user_id = str(item['user']['userId'])
    # 评论内容
    user_city = item['ipLocation']['location']
    comment = item['content'].strip().replace('\n', '').replace(',', '，')
    # 评论ID
    comment_id = str(item['commentId'])
    # 评论点赞数
    praise = str(item['likedCount'])
    # 评论时间
    date = time.localtime(int(str(item['time'])[:10]))
    date = time.strftime("%Y-%m-%d %H:%M:%S", date)
    print(user_name, user_id, user_city, comment, comment_id, praise, date)

    with open('music_comments.csv', 'a', encoding='utf-8-sig') as f:
        f.write(user_name + ',' + user_id + ',' + user_city + ',' + comment + ',' + comment_id + ',' + praise + ',' + date + '\n')
    f.close()


res = []
for data in respond["data"]["comments"]:
    res.append(
        {
            "location":data["ipLocation"],
            "text":data["content"],
        }
    )

json.dump(res, open("data/comments.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)