
import tensorflow as tf
from seqselfattention import webformerAttention
from embedding_layer import Myembeddinglayer,My_init

class MyModel(tf.keras.Model):

  def __init__(self,tagsize,T2Tattentionwidth,maxposition=4):
    super().__init__()
#    self.tagsize=tagsize
    self.embedding_layer=Myembeddinglayer(T2Tattentionwidth,tagsize,wordembedinit=My_init())#T2Tattentionwidth=6,
    self.seqselfattention1=webformerAttention(maxposition=4)
    self.seqselfattention2=webformerAttention(maxposition=4)
    self.seqselfattention3=webformerAttention(maxposition=4)
    self.seqselfattention4=webformerAttention(maxposition=4)
    self.seqselfattention5=webformerAttention(maxposition=4)
    self.seqselfattention6=webformerAttention(maxposition=4)
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


