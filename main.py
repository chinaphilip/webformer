# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:40:26 2022

@author: 1222
"""
#https://blog.csdn.net/iterate7/article/details/108959082
#这里面讲了berttokenizer的做法
#https://github.com/huggingface/transformers/blob/e6f00a11d7fa34215184e3c797e19e6c7debe0fe/src/transformers/models/bert/modeling_tf_bert.py#L715
#这页代码主要的源代码
from transformers import BertTokenizer,TFBertModel
import numpy as np 

tokenizer1 = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")#bert-base-uncased
tokenizer2 = BertTokenizer.from_pretrained("bert-base-chinese")
inputs = tokenizer2("张广寅说 Hello, my dog is cute", return_tensors="tf")

tokenizer1.vocab_size
tokenizer2.vocab_size

tokens=tokenizer1.tokenize("the game has gone!unaffable  I have a new GPU!")
print(tokens)
input_ids=tokenizer1.convert_tokens_to_ids(tokens)
print(input_ids)
print(inputs)

model = TFBertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
print(model.layers)
model.summary()
mainbertlayer=model.get_layer("bert")
mainbertlayer.summary()
len(mainbertlayer.get_weights())
mainbertlayer.get_input_embeddings()#这个返回的是一个layer
embeddinglayer=model.get_input_embeddings()#这个返回的是一个layer TFBertEmbeddings
embeddinglayer.get_weights()
dir(embeddinglayer)
embeddinglayer.variables
dir(embeddinglayer)
embeddinglayer.word_embeddings#这个返回的就是token embedding shape=(21128, 768)

#计算T2T mask矩阵，这个其实也分两步，只不过其中一步可以直接计算
testsample=[[-1,5,9,20,50,53,59],
            [-1,24,34,50,59]]#text pad后的总长度是60
onesmatrix=np.ones((2,60,60))
attention_width=5
for i in range(len(testsample)):
    for j in range(1,len(testsample[i])):
        length=testsample[i][j]-testsample[i][j-1]
        raw=np.zeros((length,length))
        for k in range(length):
            for l in range(length):
                if k-l>attention_width or l-k>attention_width:
                    raw[k][l]=1
        onesmatrix[i,(testsample[i][j-1]+1):(testsample[i][j]+1),(testsample[i][j-1]+1):(testsample[i][j]+1)]=raw


##计算H2H mask矩阵，这个还分为2步
html_edge_list
test=np.ones((2,60,60))#batchsize,htmlpadlength,htmlpadlength
#shape=np.shape(test)
for i in range(len(html_edge_list)):
    for j in range(len(html_edge_list[i])):
        test[i][html_edge_list[i][j][0]][html_edge_list[i][j][2]]=html_edge_list[i][j][1]
#计算H2H mask矩阵
equal = tf.equal(test, threshold)
c=tf.cast(c,tf.float32)

#计算H2T mask矩阵
html_text_edge_list(html_index,text_index)
test=np.ones((2,60,70))#batchsize,htmllength,textpadlength
for i in range(len(html_text_edge_list)):
    for j in range(len(html_text_edge_list[i])):
        test[i][html_edge_list[i][j][0]][html_edge_list[i][j][1]]=0

#计算T2H mask矩阵
mask=K.ones((16,32))
e=K.ones((16,40,32))
mask = K.expand_dims(K.cast(mask, K.floatx()), axis=1)#expand_dims增加向量维度，就是增加一个维度为一的维度
e=e-10000*mask
#-10000.0 * ((1.0 - mask) * (1.0 - K.permute_dimensions(mask, (0, 2, 1))))



















onesmatrix=-1*onesmatrix+1


onesmatrix=np.ones((2,2,2))


lower = K.arange(0, input_len) - self.attention_width // 2
lower = K.expand_dims(lower, axis=-1)
upper = lower + self.attention_width
indices = K.expand_dims(K.arange(0, input_len), axis=0)
attention_width_raw=(1.0 - K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx()))
        
        
        
        