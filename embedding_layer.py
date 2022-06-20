# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:58:47 2022

@author: 1222
tf.nn.embedding_lookup
tf.gather
两种实现embedding查表的方式

"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
 
class Myembeddinglayer(Layer):
    
    def __init__(self,T2Tattentionwidth,tagsize,wordembedinit,T2Tmaxposition=4, wordsize=21128,kernel_initializer='glorot_uniform',
                 pretrainedtextweight=None,output_dim=None,hiddensize=384,**kwargs):
        super().__init__()#Myembeddinglayer,self
        if output_dim==None:
            self.output_dim=hiddensize
        else:
            self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.hiddensize=hiddensize
#        super(MySelfAttention,self).__init__(**kwargs)
#        self.supports_masking = Trues
        self.pretrainedtextweight=pretrainedtextweight
        self.supports_masking = True
        self.T2Tmaxposition=T2Tmaxposition#用来计算相对位置编码的长度
        self.T2Tattentionwidth=T2Tattentionwidth
        self.wordsize=wordsize
        self.tagsize=tagsize
        self.wordembeddinit=wordembedinit
#        self.T2Tmaxposition=T2Tmaxposition
        
    def build(self,input_shape):
        self.w_html=self.add_weight(name='W_V_F',
             shape=(self.tagsize,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.w_text=self.add_weight(name='W_V_F',
             shape=(self.wordsize,768),
             initializer=self.wordembeddinit,
             trainable=True)
        self.position_embedding=self.add_weight(name='relative_position_embedding',
             shape=((self.T2Tmaxposition)*2+1,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.htmledge_embedding=self.add_weight(name='htmledge_embedding',
             shape=(3,self.hiddensize),
             initializer=self.kernel_initializer,
             trainable=True)
        self.dense1=Dense(self.hiddensize)#activation='relu'

    def call(self,inputs):
        fieldlist,textlist,htmllist,sequencelist=inputs[0],inputs[1],inputs[2],inputs[4]
        textpadlength=textlist.shape[1]
        htmlsequencelength=htmllist.shape[1]
        field_embeds = tf.gather(params=self.w_text, indices=fieldlist)
        text_embeds = tf.gather(params=self.w_text, indices=textlist)
        text_embeds=self.dense1(text_embeds)
        html_embeds = tf.gather(params=self.w_html, indices=htmllist)
        T2Tmask=self.computeT2Tmask(sequencelist,textpadlength)
        H2Hmask=self.computeH2Hmask(inputs[3][0],htmlsequencelength)#inputs[3]包含两部分，一部分是html edge list,一部分是inner text list
        H2Tmask=self.computeH2Tmask(inputs[3][1],textpadlength,htmlsequencelength)
        T2Hmask=self.computeT2Hmask(htmllist)
        htmledge_embedding_complete=tf.concat((K.ones((1,self.hiddensize)),self.htmledge_embedding),axis=0)
        return [field_embeds,text_embeds,html_embeds,T2Tmask,H2Hmask,H2Tmask,T2Hmask,self.position_embedding,htmledge_embedding_complete]


    def computeT2Tmask(self,sequencelist,textpadlength):
        #计算T2T mask矩阵，这个其实也分两步，只不过其中一步可以直接计算
#        sequencelist=[[-1,5,9,20,50,53,59],#这个最后一个数字不必是textlist的总长度
#                    [-1,24,34,50,59]]#text pad后的总长度是60
        onesmatrix=np.ones((len(sequencelist),textpadlength,textpadlength))#batchsize,textpadlength,textpadlength
        attention_width=self.T2Tattentionwidth
        for i in range(len(sequencelist)):
            for j in range(1,len(sequencelist[i])):
                length=sequencelist[i][j]-sequencelist[i][j-1]
                raw=np.zeros((length,length))
                for k in range(length):
                    for l in range(length):
                        if k-l>attention_width or l-k>attention_width:
                            raw[k][l]=1
                onesmatrix[i,(sequencelist[i][j-1]+1):(sequencelist[i][j]+1),(sequencelist[i][j-1]+1):(sequencelist[i][j]+1)]=raw
        return tf.convert_to_tensor(onesmatrix,dtype=tf.float32)
    
    
    def computeH2Hmask(self,html_edge_list,htmlsequencelength):
        ##计算H2H mask矩阵，这个还分为2步
        htmledgematrix=np.zeros((len(html_edge_list),htmlsequencelength,htmlsequencelength))#batchsize,htmlpadlength,htmlpadlength
        #shape=np.shape(test)
        for i in range(len(html_edge_list)):
            for j in range(len(html_edge_list[i])):
                htmledgematrix[i][html_edge_list[i][j][0]][html_edge_list[i][j][2]]=html_edge_list[i][j][1]
        #计算H2H mask矩阵
        #equal = tf.equal(htmledgematrix, threshold)
        #c=tf.cast(c,tf.float32)
        return tf.convert_to_tensor(htmledgematrix,dtype=tf.int32)


    def computeH2Tmask(self,html_text_edge_list,textpadlength,htmlsequencelength):
        #计算H2T mask矩阵
        #html_text_edge_list(html_index,text_index)
        H2Tmask=np.ones((len(html_text_edge_list),htmlsequencelength,textpadlength))#batchsize,htmllength,textpadlength
        for i in range(len(html_text_edge_list)):
            for j in range(len(html_text_edge_list[i])):
                H2Tmask[i][html_text_edge_list[i][j][0]][html_text_edge_list[i][j][1]]=0
        return tf.convert_to_tensor(H2Tmask,dtype=tf.float32)
    
    def computeT2Hmask(self,htmllist):
        #计算T2H mask矩阵
        #mask=K.ones((16,32))
        #e=K.ones((16,40,32))
        T2Hmask=tf.equal(tf.convert_to_tensor(htmllist,dtype=tf.int32), 0)#这个0是pad字符的index
        return T2Hmask#batchsize,htmllistlength
#        mask = K.expand_dims(K.cast(mask, K.floatx()), axis=1)#expand_dims增加向量维度，就是增加一个维度为一的维度
#        e=e-10000*mask

    def getpositionembedding(self,):
        return self.position_embedding.numpy()







#    sequencelist=[[-1,5,9,20,50,53,59],#这个最后一个数字不必是textlist的总长度
#                     [-1,24,34,50,59]]#text pad后的总长度是60

# em=Myembeddinglayer(T2Tattentionwidth=6,positionembeddinglength=4)
# em.computeT2Tmask(sequencelist,60).numpy()


class My_init(tf.keras.initializers.Initializer):

    def __init__(self,path):
        super().__init__()
        self.path=path

    def __call__(self, shape, dtype=None):
        embedding = np.load(self.path,allow_pickle=True)
        return tf.convert_to_tensor(embedding)#K.random_normal(shape, dtype=dtype)
      # return tf.random.normal(
      #     shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    # def get_config(self):  # To support serialization
    #   return {'mean': self.mean, 'stddev': self.stddev}









# class TFBertEmbeddings(tf.keras.layers.Layer):
#     """Construct the embeddings from word, position and token_type embeddings."""

#     def __init__(self, config: BertConfig, **kwargs):
#         super().__init__(**kwargs)

#         self.vocab_size = config.vocab_size
#         self.type_vocab_size = config.type_vocab_size
#         self.hidden_size = config.hidden_size
#         self.max_position_embeddings = config.max_position_embeddings
#         self.initializer_range = config.initializer_range
#         self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
#         self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

#     def build(self, input_shape: tf.TensorShape):
#         with tf.name_scope("word_embeddings"):
#             self.weight = self.add_weight(
#                 name="weight",
#                 shape=[self.vocab_size, self.hidden_size],
#                 initializer=get_initializer(self.initializer_range),
#             )

#         with tf.name_scope("token_type_embeddings"):
#             self.token_type_embeddings = self.add_weight(
#                 name="embeddings",
#                 shape=[self.type_vocab_size, self.hidden_size],
#                 initializer=get_initializer(self.initializer_range),
#             )

#         with tf.name_scope("position_embeddings"):
#             self.position_embeddings = self.add_weight(
#                 name="embeddings",
#                 shape=[self.max_position_embeddings, self.hidden_size],
#                 initializer=get_initializer(self.initializer_range),
#             )

#         super().build(input_shape)

#     def call(
#         self,
#         input_ids: tf.Tensor = None,
#         position_ids: tf.Tensor = None,
#         token_type_ids: tf.Tensor = None,
#         inputs_embeds: tf.Tensor = None,
#         past_key_values_length=0,
#         training: bool = False,
#     ) -> tf.Tensor:
#         """
#         Applies embedding based on inputs tensor.
#         Returns:
#             final_embeddings (`tf.Tensor`): output embedding tensor.
#         """
#         if input_ids is None and inputs_embeds is None:
#             raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

#         if input_ids is not None:
#             inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

#         input_shape = shape_list(inputs_embeds)[:-1]

#         if token_type_ids is None:
#             token_type_ids = tf.fill(dims=input_shape, value=0)

#         if position_ids is None:
#             position_ids = tf.expand_dims(
#                 tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
#             )

#         position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
#         token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
#         final_embeddings = inputs_embeds + position_embeds + token_type_embeds
#         final_embeddings = self.LayerNorm(inputs=final_embeddings)
#         final_embeddings = self.dropout(inputs=final_embeddings, training=training)

#         return final_embeddings

# """
