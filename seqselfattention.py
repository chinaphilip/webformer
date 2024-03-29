# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:56:27 2022

@author: 1222
"""

from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
 
class webformerAttention(Layer):
    
    def __init__(self,maxposition=6,output_dim=None,kernel_initializer='glorot_uniform',hiddensize=384,**kwargs):
        super().__init__()#webformerAttention,self
        if output_dim==None:
            self.output_dim=hiddensize
        else:
            self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.hiddensize=hiddensize
#        super(MySelfAttention,self).__init__(**kwargs)
        self.supports_masking = True
        self.max_position=maxposition
        
        
    def build(self,input_shape):#input_shape((batchsize,sequencelength,hidden size))
        # self.W=self.add_weight(name='W',
        #      shape=(3,input_shape[2],self.output_dim),
        #      initializer=self.kernel_initializer,
        #      trainable=True)
        self.W_V_F=self.add_weight(name='W_V_F',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_V_H=self.add_weight(name='W_V_H',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_V_T=self.add_weight(name='W_V_T',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_Q_H2T=self.add_weight(name='W_Q_H2T',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_Q_T2T=self.add_weight(name='W_Q_T2T',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_Q_T2H=self.add_weight(name='W_Q_T2H',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_Q_H2H=self.add_weight(name='W_Q_H2H',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_K_T2T=self.add_weight(name='W_K_T2T',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_K_T2H=self.add_weight(name='W_K_T2H',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_K_H2T=self.add_weight(name='W_K_H2T',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.W_K_H2H=self.add_weight(name='W_K_H2H',
             shape=(self.hiddensize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.built = True
        
    def call(self,inputs,final_layer=False,**kwargs):

        field, text, html= inputs[0], inputs[1], inputs[2], 
        positionembedding=inputs[7]
        H2Hmask=inputs[4]
        htmledgeembedding=inputs[8]#batchsize htmllength htmllength
        H2Tmask=inputs[5]
        T2Tmask=inputs[3]
        T2Hmask=inputs[6]
        textlength=text.shape[1]
        
        
        text_v=K.dot(text,self.W_V_T)
        html_v=K.dot(html,self.W_V_H)
        #H2H attention
        H2H_q=K.dot(html,self.W_Q_H2H)
        #H2H_k=K.dot(html,self.W_K_H2H)
        H2H_k=tf.add(tf.expand_dims(K.dot(html,self.W_K_H2H),1),K.gather(htmledgeembedding, H2Hmask))

        #H2H_e=K.batch_dot(H2H_q,K.permute_dimensions(H2H_k,[0,2,1]))#
        H2H_e=tf.reduce_sum(tf.multiply(tf.expand_dims(H2H_q,1),H2H_k),axis=-1)
        #H2T attention
        H2T_q=K.dot(html,self.W_Q_H2T)
        H2T_k=K.dot(text,self.W_K_H2T)
        H2T_e=K.batch_dot(H2T_q,K.permute_dimensions(H2T_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同
        #T2H attention
        T2H_q=K.dot(text,self.W_Q_T2H)
        T2H_k=K.dot(html,self.W_K_T2H)
        T2H_e=K.batch_dot(T2H_q,K.permute_dimensions(T2H_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同
        #T2T attention
        T2T_q=K.dot(text,self.W_Q_T2T)
        pos_ids=self.cal_rel_pos_matri(textlength)
        T2T_k=tf.add(tf.expand_dims(K.dot(text,self.W_K_T2T),1),tf.expand_dims(K.gather(positionembedding, pos_ids), 0))
        T2T_e=tf.reduce_sum(tf.multiply(tf.expand_dims(T2T_q,1),T2T_k),axis=-1)
        #T2T_e=K.batch_dot(T2T_q,K.permute_dimensions(T2T_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同
        
        """
        batch_dot
          if axes is None:
            if y_ndim == 2:
              axes = [x_ndim - 1, y_ndim - 1]
            else:
              axes = [x_ndim - 1, y_ndim - 2]
        """
        H2H_e=H2H_e/(self.output_dim**0.5)
        H2T_e=H2T_e/(self.output_dim**0.5)
        T2H_e=T2H_e/(self.output_dim**0.5)
        T2T_e=T2T_e/(self.output_dim**0.5)

        H2Hmask_2=tf.equal(H2Hmask,0)#0代表两个tag之间没有边连接
        H2Hmask_2=tf.cast(H2Hmask_2,tf.float32)
        
        #H2H_e-=10000(1-K.cast(mask[0], K.floatx())
        H2H_e-=10000*H2Hmask_2#(batchsize,htmllength,textlength)
        
        #H2T_e-=10000(1-K.cast(mask[1], K.floatx())
        H2T_e-=10000*H2Tmask#(batchsize,htmllength,textlength)
        #T2T_e-=10000(1-K.cast(mask[2], K.floatx())
        T2T_e-=10000*T2Tmask#textsequencematrix(batchsize,textlength,textlength)


        H2H_e=K.softmax(H2H_e)
#        print(K.int_shape(H2H_e))
        html1=K.batch_dot(H2H_e,html_v)

        H2T_e=K.softmax(H2T_e)
        html2=K.batch_dot(H2T_e,text_v)

        T2H_e=K.softmax(T2H_e)
        text1=K.batch_dot(T2H_e,html_v)

        T2T_e=K.softmax(T2T_e)
        text2=K.batch_dot(T2T_e,text_v)
        
        html=html1+html2
        text=text1+text2
        if final_layer==False:
            return [field,text,html,inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8]]
        else:
            return [field,text,html]
    


    # def compute_mask(self, inputs, mask=None):
    #     if isinstance(mask, list):
    #         mask = mask[0]
    #     if self.return_attention:
    #         return [mask, None]
    #     return mask
    
    # def compute_output_shape(self,input_shape):
    #     return (input_shape[0],input_shape[1],self.output_dim)

    def cal_rel_pos_matri(self,textlength):
        # batchsequencelist=inputs[1],inputs[4]#数据以batch的形式输入
        # batchsize,textlength=np.shape(batchtextlist)
        # textrealtiveposition=tf.zeros(batchsize,textlength,textlength)
        # for sample in batchsequencelist:
        #     for j in sample:
        #         tf.zeros

        # 计算相对位置矩阵位置差
        # 一维[0,1,...,q_seq_len]
        q_idxs = K.arange(0, textlength, dtype='int32')
        # [[0] [1] [2] [3] ... [q_seq_len]]
        q_idxs = K.expand_dims(q_idxs, 1)
        v_idxs = K.arange(0, textlength, dtype='int32')
        # [[0,1,...,v_seq_len]]
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        '''以q_seq_len=v_seq_len=9为例：
         [[ 0  1  2  3  4  5  6  7  8  9]
          [-1  0  1  2  3  4  5  6  7  8]
          [-2 -1  0  1  2  3  4  5  6  7]
          [-3 -2 -1  0  1  2  3  4  5  6]
          [-4 -3 -2 -1  0  1  2  3  4  5]
          [-5 -4 -3 -2 -1  0  1  2  3  4]
          [-6 -5 -4 -3 -2 -1  0  1  2  3]
          [-7 -6 -5 -4 -3 -2 -1  0  1  2]
          [-8 -7 -6 -5 -4 -3 -2 -1  0  1]
          [-9 -8 -7 -6 -5 -4 -3 -2 -1  0]]
          相对位置编码就比较简单的用这种差几位数来表示相对位置
        '''
        # 后处理操作
        #max_position =self.maxposition #(self.input_dim - 1) // 2
        '''
        K.clip：逐元素clip，将pos_ids中超出(-max_position, max_position)范围的数强制变为边界值
        1、作者假设精确的相对位置编码在超出了一定距离之后是没有必要的
        2、截断最大距离使得模型的泛化效果好，可以更好的generalize到没有在训练阶段出现过的序列长度上
        比如上面的例子中, 截到(-4,4)之间为：
        [[ 0  1  2  3  4  4  4  4  4  4]
         [-1  0  1  2  3  4  4  4  4  4]
         [-2 -1  0  1  2  3  4  4  4  4]
         [-3 -2 -1  0  1  2  3  4  4  4]
         [-4 -3 -2 -1  0  1  2  3  4  4]
         [-4 -4 -3 -2 -1  0  1  2  3  4]
         [-4 -4 -4 -3 -2 -1  0  1  2  3]
         [-4 -4 -4 -4 -3 -2 -1  0  1  2]
         [-4 -4 -4 -4 -4 -3 -2 -1  0  1]
         [-4 -4 -4 -4 -4 -4 -3 -2 -1  0]]
        '''
        pos_ids = K.clip(pos_ids, -self.max_position,
                         self.max_position)
        pos_ids = pos_ids + self.max_position  # shape=(q_seq_lenv, v_seq_len)
        return pos_ids











# text=K.ones((16,500,768))
# html=K.ones((500,500,768))
# text=K.random_normal((4,10,20))
# html=K.random_normal((10,10,20))
# html = tf.expand_dims(html, 0)
# text = tf.expand_dims(text, 1)
# result=tf.add(text,html)
# result[0,0,1,:]==text[0,0,1,:]+html[0,0,1,:]



# test=tf.convert_to_tensor([[1,1,1,1,1,0,0,0,0],
#                       [1,1,1,1,1,1,1,0,0],
#                       [1,1,1,1,1,0,0,0,0]])
# c=tf.cast(tf.equal(test, 0),tf.float32)





# another=K.placeholder((16,1,500,768))
# another2=K.placeholder((16,500,500,768))

# another=K.ones((16,500,768))
# another = tf.expand_dims(another, 2)
# field=tf.ones((32,5,768))
# testattention=MySelfAttention()
# testattention(inputs=[field,html,text])
# K.shape(tf.reduce_sum(tf.multiply(another,another2,),axis=-1)#tf.squeeze(axes=(-1,-1)
# K.shape(tf.einsum("mnp,onp->mon",text,html))
# result=tf.einsum("mnp,onp->mon",text,html)

# result[0,0,0]==K.dot(text[0,0,:],html[0,0,:])



# test1=K.placeholder((3,768))
# test2=K.placeholder((1,768))
# test3=K.placeholder((16,500,500),dtype="int32")
# test4=tf.add(tf.expand_dims(test1,1),K.gather(test2, test3))
# test5=K.placeholder((16,500,768))
# tf.reduce_sum(tf.multiply(tf.expand_dims(test5,1),test4),axis=-1)



# tf.concat([test1,test2], axis=0, name='concat')：
