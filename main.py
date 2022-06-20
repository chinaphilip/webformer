import importlib
from lxml import etree
from lxml import html
from itertools import combinations,permutations
#from graphviz import Digraph
from transformers import BertTokenizer,TFBertModel
import numpy as np
import os
from utils import getlist,padlistf
from model import webformer
from seqselfattention import webformerAttention
from embedding_layer import Myembeddinglayer,My_init

#初始化分词器
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

#加载htmltag字典
htmltagpath=r"G:\ner\webformer implementation\html_tag_list.txt"
with open(htmltagpath, "r",encoding='utf-8') as f:
    taglist=f.readlines()# 打开文件
taglist=[j[1:-3] for j in taglist]
tagdict={}
tagdict["<pad>"]=0
for index,tag in enumerate(taglist):
    tagdict[tag]=index+1


#准备测试样本
bfieldlist=[]#这个是要抽取的属性集合
bhtmllist=[]#这个是网页的html节点list
btextlist=[]#这个是网页的字符list
bhtmledgelist=[]#[html1,edgetype,html2]具体表示方式可以在函数traversehtml中看到
binnertextlist=[]#[html,text]#这个存储tag和text之间的关系
btextsequencelist=[]#这个存储text和text之间的关系

path="G:\lecture_spider_data\shangcai_"#这里放样本数据集的path

for root, dirs, files in os.walk(path):
    for f in files:
        if f.split(".")[1]!="html":
            continue
        with open(os.path.join(root, f), "r",encoding='utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件
            html = etree.HTML(data)
            output=getlist(html,tagdict,tokenizer)
            bhtmllist.append(output[0])
            btextlist.append(output[1])
            bhtmledgelist.append(output[2])
            binnertextlist.append(output[3])
            btextsequencelist.append(output[4])

bhtmllist=padlistf(bhtmllist,0)
btextlist=padlistf(btextlist,0)
bfieldlist=["时间"]*(bhtmllist.shape[0])#这个field就是指要抽取的属性名
#tokens=tokenizer.tokenize(tranuctedtext)
#input_ids=tokenizer.convert_tokens_to_ids(tokens)
bfieldlist=[[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i))] for i in bfieldlist]
bfieldlist=np.array(bfieldlist)

#测试模型是否能跑通，输出是否正确
model = webformer(tagsize=119,T2Tattentionwidth=4,word_embed_path=r"G:\ner\webformer implementation\bert_word_embedding.dat")#这个embedding文件有提供下载方式
output=model([bfieldlist[:2],btextlist[:2],bhtmllist[:2],[bhtmledgelist[:2],binnertextlist[:2]],btextsequencelist[:2]])#batchsize设为2，再高一点他就爆内存了
model.summary()
model.get_layer()
