# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:56:27 2022

@author: 1222
"""

from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
 
class MySelfAttention(Layer):
    
    def __init__(self,local_radius=None,output_dim=None,kernel_initializer='glorot_uniform',hiddensize=768,**kwargs):
        if output_dim==None:
            self.output_dim=hiddensize
        else:
            self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.hiddensize=hiddensize
        super(MySelfAttention,self).__init__(**kwargs)
        self.supports_masking = True
        self.local_radius=local_radius
        
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
        
    def call(self,inputs, mask=None, **kwargs):
        
        field, html, text = inputs[0], inputs[1], inputs[2]
        text_v=K.dot(text,self.W_V_T)
        html_v=K.dot(html,self.W_V_H)
        #H2H attention
        H2H_q=K.dot(html,self.W_Q_H2H)#x(batchsize,sequencelength,hidden size)(hiddensize, hiddensize)(batchsize,sequencelength,hidden size)
        H2H_k=K.dot(html,self.W_K_H2H)
        H2H_e=K.batch_dot(H2H_q,K.permute_dimensions(H2H_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同#(batchsize,sequencelength,hidden size)(batchsize,hidden size,sequencelength)(batchsize,sequencelength,sequencelength)
        #H2T attention
        H2T_q=K.dot(html,self.W_Q_H2T)#x(batchsize,sequencelength,hidden size)(hiddensize, hiddensize)(batchsize,sequencelength,hidden size)
        H2T_k=K.dot(text,self.W_K_H2T)
        H2T_e=K.batch_dot(H2T_q,K.permute_dimensions(H2T_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同
        #T2H attention
        T2H_q=K.dot(text,self.W_Q_T2H)#x(batchsize,sequencelength,hidden size)(hiddensize, hiddensize)(batchsize,sequencelength,hidden size)
        T2H_k=K.dot(html,self.W_K_T2H)
        T2H_e=K.batch_dot(T2H_q,K.permute_dimensions(T2H_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同
        #T2T attention
        T2T_q=K.dot(text,self.W_Q_T2T)#x(batchsize,sequencelength,hidden size)(hiddensize, hiddensize)(batchsize,sequencelength,hidden size)
        T2T_k=K.dot(text,self.W_K_T2T)
        T2T_e=K.batch_dot(T2T_q,K.permute_dimensions(T2T_k,[0,2,1]))#把k转置，并与q点乘,dot的维度长度必须相同
        
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
        #mask[0]H2H mask
        #mask[1]H2T mask
        #mask[2]T2T mask
        H2H_e-=10000(1-K.cast(mask[0], K.floatx())
        H2T_e-=10000(1-K.cast(mask[1], K.floatx())
        T2T_e-=10000(1-K.cast(mask[2], K.floatx())
        
        
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
    
    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)



html = tf.ones((32,500,768))
text=tf.ones((32,600,768))
field=tf.ones((32,5,768))
testattention=MySelfAttention()
testattention(inputs=[field,html,text])
