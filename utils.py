import numpy as np
from bs4 import BeautifulSoup,Comment,NavigableString,Comment
from lxml import html
from itertools import combinations,permutations

def travesehtml(presentnode, htmllist, textlist, htmledgelist, innertextlist, textsequencelist,tagdict,tokenizer):
    """
    这个函数是个递归函数，主要是用来遍历一个网页，获取其中所有的tag节点，文本节点以及各种边
    Args:
        presentnode:这个是一个lxml网页节点对象
        htmllist: 存储网页所有的tag节点
        textlist: 存储网页所有的字符
        htmledgelist: 存储tag节点之间的关系,存储方式如下,index为该节点在htmllist中的位置
            1   parent [parent node index, 1, child node index]
            2   child  [parent node index, 2, child node index]
            3   sibling [parent node index, 3, child node index]
        innertextlist:这个存储的是tag节点与字符之间的关系，格式为[tag_index,word_index]
        textsequencelist:存储字符之间的关系，就是任意两个字符属不属于同一个textnode，关系表示的方式是存储textlist的划分标记[first_textnode_length,second_textnode_length....]

    Returns:

    """
    #    htmllist.append(presentnode.tag)
    presentnodeposition = len(htmllist) - 1
    presentchildren = presentnode.getchildren()

    if presentnode.text != None and presentnode.text.strip() != "":
        text = presentnode.text.strip()
        if len(text) > 200:
            tranuctedtext = text[:100] + text[-100:]#过长的文本节点只取前100个字符和后100个字符
        else:
            tranuctedtext = text
        tokens = tokenizer.tokenize(tranuctedtext)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        textstartposition = len(textlist)
        textlist.extend(input_ids)
        textendposition = len(textlist)
        textsequencelist.append(len(input_ids) + textsequencelist[-1])
        for i in range(textstartposition, textendposition):
            innertextlist.append([presentnodeposition, i])
    # 第一个位置放置父元素的位置
    childrenpositionlist = []
    for i in range(len(presentchildren)):
        if presentchildren[i].tag in tagdict:
            htmllist.append(tagdict[presentchildren[i].tag])
        else:

            tagdict[presentchildren[i].tag] = len(tagdict)
            htmllist.append(tagdict[presentchildren[i].tag])
        childrenposition = len(htmllist) - 1
        childrenpositionlist.append(childrenposition)
        htmledgelist.append([presentnodeposition, 1, childrenposition])
        htmledgelist.append([childrenposition, 2, presentnodeposition, ])
        #        a.append((presentchildren[i],0))
        #        positionlist.app
        travesehtml(presentchildren[i], htmllist, textlist, htmledgelist, innertextlist, textsequencelist,tagdict,tokenizer)
    # for j in range(1,len(positionlist)):
    #     b.append((positionlist[0],-1,positionlist[j]))
    #     b.append((positionlist[j],-2,positionlist[0]))
    d = list(combinations(childrenpositionlist, 2))
    for k in range(len(d)):
        htmledgelist.append([d[k][0], 3, d[k][1]])




def getlist(html,tagdict,tokenizer):
    #这个函数就是单纯调用travesehtml
    htmllist=[tagdict["html"]]
    textlist=[]
    htmledgelist=[]#[html1,edgetype,html2]
    innertextlist=[]#[html,text]
    textsequencelist=[-1,]
    travesehtml(html,htmllist,textlist,htmledgelist,innertextlist,textsequencelist,tagdict,tokenizer)
    return [htmllist,textlist,htmledgelist,innertextlist,textsequencelist]

def padlistf(nestedlist,padnumber):
    #给各种batchlist做padding
    #输入的是嵌套list
    maxlength=0
    for i in nestedlist:
        if len(i)>maxlength:
            maxlength=len(i)
    for i in range(len(nestedlist)):
        padlist=[padnumber]*(maxlength-len(nestedlist[i]))
        nestedlist[i].extend(padlist)
    return np.array(nestedlist,dtype=np.int32)