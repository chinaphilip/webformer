# webformer
目前的实现方式有一点问题，我的输入样本中每个网页的字符长度大概是1000左右，在计算attention的时候，relative position embedding矩阵会特别的大，会爆内存

模型主要是包含三个文件

embeddinglayer.py 这个文件定义了网页标记符和字符的embedding,计算了下面的attention layer中需要用到的四种mask

model.py 定义了模型的整体结构，最后的输出只是encoder representation，没有接后端任务

seqselfattention.py 定义了论文中提到的四种attention

运行方式：打开main.py从上往下运行即可

bert_word_embedding.dat文件下载链接
链接：https://pan.baidu.com/s/1yg4mHFKE2KmCHiwhFG0z7w?pwd=dbjx 
提取码：dbjx

