---
<!-- 该部分为参数配置部分 -->

---
<!-- 公共内容部分  -->

## 模型加载和推理
更多关于模型加载和推理的问题参考[模型的推理Pipeline](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline)。

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')
```

提供input输入
```python
semantic_cls(input='新年快乐！')
```

更多使用说明请参阅[ModelScope文档中心](http://www.modelscope.cn/#/docs)。

---
<!-- 在线使用独有内容部分 -->

---
<!-- 本地使用独有内容部分 -->
## 下载并安装ModelScope library
更多关于下载安装ModelScope library的问题参考[环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。

```python
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
