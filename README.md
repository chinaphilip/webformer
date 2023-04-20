# webformer：The Web-page Transformer for Structure Information Extraction
https://arxiv.org/abs/2202.00217
![avatar](/webformer.PNG)

This paper try to reimplement the webformer. The author has share part of the core code to me. If interested, you can request this piece of code through my email aauiuui@163.com. The code in this repository is just the manuscript, which can not be directly put into application.

这个仓库是试图复现上面的这篇论文
目前的实现方式有一点问题，我的输入样本中每个网页的字符长度大概是1000左右，在计算attention的时候，relative position embedding矩阵会特别的大，会爆内存
感兴趣的可以去参考bigbert的实现代码，原作者分享了一部分源码，这里不方便直接放出，感兴趣的可以邮件联系我索取

模型主要是包含三个文件

embeddinglayer.py 这个文件定义了网页标记符和字符的embedding,计算了下面的attention layer中需要用到的四种mask

model.py 定义了模型的整体结构，最后的输出只是encoder representation，没有接后端任务

seqselfattention.py 定义了论文中提到的四种attention

运行方式：打开main.py从上往下运行即可

bert_word_embedding.dat文件下载链接
链接：https://pan.baidu.com/s/1Rxei5cG3q45Hts3YpOhnCw?pwd=nb8u 
提取码：nb8u

