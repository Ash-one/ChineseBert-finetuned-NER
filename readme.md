## 概述

本项目为基于fastNLP/pytorch实现Bert+BiLSTM+CRF的命名实体识别任务，并用flask实现服务端可视化展示。

![结果展示](https://s2.loli.net/2023/03/25/e53mHihFl8JENqV.png)

由于模型过大所以需要单独下载。百度云链接: https://pan.baidu.com/s/1X43pDB0Acw7lqsvkHNknwA?pwd=wm6e

jupyter notebook中是模型的训练过程和一些工具函数，都可以运行。

其他细节可以通过这两篇博客查看：

- 1 [使用Bert进行中文NER命名实体识别feat.fastNLP（上：模型篇）](https://ash-one.github.io/2023/03/18/shi-yong-bert-jin-xing-ner-ming-ming-shi-ti-shi-bie-feat-fastnlp/)

- 2 [使用Bert进行NER命名实体识别feat.fastNLP(下：使用flask部署模型)](https://ash-one.github.io/2023/03/25/shi-yong-bert-jin-xing-ner-ming-ming-shi-ti-shi-bie-feat-fastnlp-xia-shi-yong-flask-bu-shu-mo-xing/)

## 运行
服务器上运行flask服务端，需要运行`run.py`文件。
