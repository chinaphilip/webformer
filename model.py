# -*- coding: utf-8 -*-
"""
Created on Mon May  2 03:50:01 2022

@author: 1222
"""

import tensorflow as tf
from seqselfattention import webformerAttention
from embedding_layer import Myembeddinglayer

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.embedding_layer=Myembeddinglayer(T2Tattentionwidth=6,positionembeddinglength=4)
    self.seqselfattention1=webformerAttention()
    self.seqselfattention2=webformerAttention()
    self.seqselfattention3=webformerAttention()
    self.seqselfattention4=webformerAttention()
    self.seqselfattention5=webformerAttention()
    self.seqselfattention6=webformerAttention()
#    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
      x = self.embedding_layer(inputs)
      x=self.seqselfattention1(x)
      x=self.seqselfattention2(x)
      x=self.seqselfattention3(x)
      x=self.seqselfattention4(x)
      x=self.seqselfattention5(x)
      x=self.seqselfattention6(x,final_layer=True)
      
      return x









model = MyModel()
model.get_layer
model.summary()