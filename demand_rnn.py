import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import datetime
import tensorflow as tf

class Config(object):
    """Holds default parameters and hyperparameters for networ
    """
    Nloc=1
    Nstation=1
    Nin = Nloc+Nstation+9
    Nhid=Nin
    Nout=1
    dropout=0.5
    lr = 0.01
    n_epochs=100

    
#Try building LSTM recurrent neural network, with single hidden layer.    
class demand_model(model):
    def add_placeholders(self,dep_vars,times):
        """add_placeholders
        Make placeholder variables for the inputs. 

        Current demand
        Temperature
        Day of the week (one-hot)
        """
        self.demand       = tf.placeholder(tf.float32,shape=(Nloc,))
        self.temp         = tf.placeholder(tf.float32,shape=(Nstation,))
        self.day_of_week  = tf.placeholder(tf.float32,shape=(7,))
        self.time_of_day  = tf.placeholder(tf.float32)
        self.frac_of_year = tf.placeholder(tf.float32)
        
    def create_feed_dict(self,demand,temp,day_of_week,time_of_day,frac_of_year):
        """create_feed_dict
        Pass actual values via a dictionary to the input placeholders.
        """
        feed_dict={self.demand:demand,
                   self.temp:temp,
                   self.day_of_week:day_of_week,
                   self.time_of_day:time_of_day,
                   self.frac_of_year:frac_of_year}
        return feed_dict
        
    def add_prediction_op:
        """add_prediction_op
        Define the network to be optimized from inputs to outputs.
        """
        xin = 

        #Weight matrices from inputs to internal, forget, output
        Wix = tf.Variable(xavier_initializer([Config.Nhid,Config.Nin]))
        Wfx = tf.Variable(xavier_initializer([Config.Nhid,Config.Nin]))        
        Wox = tf.Variable(xavier_initializer([Config.Nout,Config.Nin]))

        #matrices from hidden layer
        Wih = tf.Variable(xavier_initializer([Config.Nhid,Config.Nout]))
        Wfh = tf.Variable(xavier_initializer([Config.Nhid,Config.Nout]))        
        Woh = tf.Variable(xavier_initializer([Config.Nout,Config.Nout]))
        
        bi = tf.Variable(tf.ones([Config.Nhid,]))
        bf = tf.Variable(tf.ones([Config.Nhid,]))
        bh = tf.Variable(tf.ones([Config.Nout,]))

        h=tf.nn.relu(tf.matmul(x,W)+b1)
        h_drop=tf.nn.dropout(h,keep_prob=self.dropout_placeholder)
        pred=tf.matmul(h_drop,U)+b2
        #pred=tf.Print(pred,[pred],summarize=15)
        
        ### END YOUR CODE


        
        
    def add_loss_op:
        """add_loss_op
        Define the loss/cost function to optimize.
        """

    def add_training_op:
        """add_training_op
        Define the training method (optimizer) to use.  
        """

        
