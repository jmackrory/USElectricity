"""
Recurent Neural Network

Creates python class for creating, training, saving, infering a
Tensorflow graph to forecast time-series. 
Makes a simple multilayer RNN, with mapping from inputs 
to hidden layers.  

Based on object oriented framework used in CS 224,
and A. Geron's "Hands on Machine Learning with Scikit-Learn and
Tensorflow" Ch 14 on RNN.  
The tensorflow docs are pretty rough, but the tutorials are almost readable. 

The input (X), and target (y) placeholders are defined in add_placeholders.
These are inputs to the TF graph.

This takes in inputs from Nstep_in.
This is mapped down to a different number of dimensions via a linear tranformation layer.
There is then a multi-layer RNN with dropout for regularization.
Then the model has a final linear layer (analogous to attention?)
to output a sequence of length Nstep_out, with Noutput variables at each step. 

Before the network can be run it should be built, which defines the 
The guts of the network are defined in add_prediction_op, which
has an input/output hidden layer to reduce dimension.  
There is then a multilayer, dynamic RNN inside.
This is all defined with tensorflow intrinsics/added modules.
Currently, I've turned off the dropout, which should only be active
during training.  (Can toggle this by tweeking keep_prob).

The training is done with batch gradient descent optimization
via the Adam optimizer.

The loss/cost function is defined in add_loss_op, and is just 
the mean-square error across inputs.

Prediction and inference is done in predict_all()
In order to do prediction/inference, a model is loaded from a saved file
(with graph defined in a Metagraph, and variables loaded via Saver).

Currently data is read in via feed_dict, which is super slow.
Apparently tf.Data is the new preferred simple framework for this.
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, GRUCell, LSTMCell,\
    DropoutWrapper

import numpy as np
import matplotlib.pyplot as plt

#to prevent creating huge logs.
from IPython.display import clear_output
import time

class RNNConfig(object):
    """
    RNNConfig

    Storage class for Recurrent network parameters.

    """
    def __init__(self,Nsteps_in=24, Nsteps_out=24,
                 Ninputs=1, Noutputs=1,
                 cell='RNN',Nlayers=2,Nhidden=100,
                 lr=0.001,Nepoch=1000,keep_prob=0.5):
        #number of outputs per input
        self.Noutputs=Noutputs
        self.Ninputs=Ninputs        
        #number of steps
        self.Nsteps_in=Nsteps_in
        self.Nsteps_out=Nsteps_out
        #number of dim on input
        self.cell_type=cell
        self.Nlayers=Nlayers
        self.Nhidden=Nhidden
        self.lr = lr
        self.keep_prob=0.5
        self.Nepoch=Nepoch
        self.Nprint=20
        #only grabbing a fraction of the data
        self.Nbatch=100

    def __print__(self):
        return self.__dict__

class NNModel(object):
    """
    Abstract Base Class for Tensorflow Neural Network Model.
    Designed with time-series in mind.  
    """
    def __init__(self,config):
        """
        Initialize model and build initial graph.
        config - instance of NNModel with parameters.
        """
        self.config=config
        self.keep_prob=self.config.keep_prob
        #makes the tensor flow graph.
        
        self.build()

    def build(self):
        """Creates essential components for graph, and 
        adds variables to instance. 
        """
        tf.get_default_graph()
        tf.reset_default_graph()        
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """Adds input, output placeholders to graph. 
        Note that these are python object attributes.
        """
        #load in the training examples, and their labels
        #inputs:  Nobs, with n_steps, and n_inputs per step
        self.X = tf.placeholder(tf.float32,[None,self.config.Nsteps_in,self.config.Ninputs],name='X')
        #Outputs: n_outputs we want to predict in the future.
        self.y = tf.placeholder(tf.float32,[None,self.config.Nsteps_out,self.config.Noutputs],name='y')


    #should really fix to use dataset, with batch generator.    
    def create_feed_dict(self,inputs_batch, labels_batch=None):
        """Make a feed_dict from inputs, labels as inputs for 
        graph.
        Args:
        inputs_batch - batch of input data
        label_batch  - batch of output labels. (Can be none for prediction)
        Return:
        Feed_dict - the mapping from data to placeholders.
        """
        feed_dict={self.X:inputs_batch}
        if labels_batch is not None:
            feed_dict[self.y]=labels_batch
        return feed_dict

    
    def add_prediction_op(self):
        """The core model to the graph, that
        transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        
        Returns: outputs - tensor

        """
        raise NotImplementedError
        return outputs

    def add_loss_op(self,outputs):
        """Add ops for loss to graph.
        Uses mean-square error as the loss.  (Nice, differentiable)
        """
        loss = tf.reduce_mean(tf.square(outputs-self.y))                
        return loss

    def add_training_op(self,loss):
        """Create op for optimizing loss function.
        Can be passed to sess.run() to train the model.
        Args:
           loss - TF scalar variable for loss over batch(e.g. RMSE)
        Return: 
          training_op - operation to train and update graph
        """
        optimizer=tf.train.AdamOptimizer(learning_rate=self.config.lr)
        training_op=optimizer.minimize(loss)
        return training_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: current tensorflow session
            input_batch:  np.ndarray of shape (Nbatch, Nfeatures)
            labels_batch: np.ndarray of shape (Nbatch, 1)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        #_, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        sess.run(self.train_op, feed_dict=feed)
        loss=sess.run(self.loss, feed_dict=feed)                
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: current tensorflow session
            input_batch: input data np.ndarray of shape (Nbatch, Nstep,Nfeatures)
        Returns:
            predictions: np.ndarray of shape (Nbatch, Nout,Nfeatures,)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    # #Should use tf.Data as described in seq2seq.
    #Much faster than feed_dict according to TF docs
    
    def get_random_batch(self,X,y):
        """get_random_batch(X,y)   
        Gets multiple random samples for the data.
        Makes list of returned entries.
        Then combines together with 'stack' function at the end.
        Currently selected the next days change in stock price.

        X - matrix of inputs, (Nt, Ninputs)
        Y - matrix of desired outputs (Nt,Noutputs)

        Outputs:
        X_batch - random subset of inputs shape (Nbatch,Nsteps,Ninputs) 
        y_batch - corresponding subset of outputs (Nbatch,Nsteps)
        """
        Nt,Nin = X.shape
        x_list=[]
        y_list=[]
        for i in range(self.config.Nbatch):
            #find starting time for inputs
            n0 =int(np.random.random()*(Nt-self.config.Nsteps_in-self.config.Nsteps_out))
            #edge of input data / start-point of target
            n1 = n0+self.config.Nsteps_in
            #end-point of target
            n2 = n1+self.config.Nsteps_out
            x_sub = X[n0:n1]
            y_sub = y[n1:n2]
            x_list.append(x_sub)
            y_list.append(y_sub)
        x_batch=np.stack(x_list,axis=0)
        y_batch=np.stack(y_list,axis=0)
        return x_batch,y_batch
    
    def train_graph(self,Xi,yi,save_name=None):
        """train_graph
        Actually trains the graph, and saves it. o

        Args: Xi - input data
              yi - target output data 
              save_name - path to use to save Models under.
        Side effect: Prints output loss, and makes some plots of error, and logs.
                     
        """
        self.is_training=True
        #distinguish between stored value for keep, and training/test value. 
        self.keep_prob=self.config.keep_prob
        #save model and graph
        saver=tf.train.Saver()
        init=tf.global_variables_initializer()
        loss_tot=np.zeros(int(self.config.Nepoch/self.config.Nprint+1))
        #Try adding everything by name to a collection
        tf.add_to_collection('X',self.X)
        tf.add_to_collection('y',self.y)
        tf.add_to_collection('loss',self.loss)
        tf.add_to_collection('pred',self.pred)
        tf.add_to_collection('train',self.train_op)
        
        with tf.Session() as sess:
            init.run()
            # if (save_name!=None):
            #     saver.save(sess,save_name)
            t0=time.time()
            #Use Writer for tensorboard.
            writer=tf.summary.FileWriter("logdir-train",sess.graph)            
            for iteration in range(self.config.Nepoch+1):
                #select random starting point.
                X_batch,y_batch=self.get_random_batch(Xi,yi)
                current_loss=self.train_on_batch(sess, X_batch, y_batch)
                t2_b=time.time()
                if (iteration)%self.config.Nprint ==0:
                    clear_output(wait=True)
                    print('iter #{}. Current MSE:{}'.format(iteration,current_loss))
                    print('Total Time taken:{}'.format(t2_b-t0))
                    print('\n')
                    #save the weights
                    if (save_name != None):
                        saver.save(sess,save_name,global_step=iteration,
                                   write_meta_graph=True)
                    #manual logging of loss    
                    loss_tot[int(iteration/self.config.Nprint)]=current_loss
            writer.close()
            #Manual plotting of loss.  Writer/Tensorboard supercedes this .
            plt.figure()                            
            plt.plot(loss_tot)
            plt.ylabel('Error')
            plt.xlabel('Iterations x{}'.format(self.config.Nprint))
            plt.show()

    def predict_all(self,input_data,model_name,num=None,reset=False):
        """predict_all
        Load a saved Neural network, and predict the output labels
        based on input_data.  Predicts the whole sequence, using
        the batching to process the data in sequence. 
        
        Note this only uses a single prediction for each time.
    
        Input: input_data - transformed data of shape (Nobs,Nfeature).
        model_name - name to where model/variables are saved.
        num - number of iteration to use.  (defaults to number of epochs)
        reset - optional flag to force a reset of default TF graph.

        Output: nn_pred_reduced - vector of predicted labels.
        """
        if (reset):
            tf.reset_default_graph()        
        self.is_training=False
        self.keep_prob=1
        if (num==None):
            full_model_name=model_name+'-{}'.format(self.config.Nepoch)
        else:
            full_model_name=model_name+'-{}'.format(num)           
        with tf.Session() as sess:
            saver=tf.train.import_meta_graph(full_model_name+'.meta')
            #restore graph structure
            self.X=tf.get_collection('X')[0]
            self.y=tf.get_collection('y')[0]
            self.pred=tf.get_collection('pred')[0]
            self.train_op=tf.get_collection('train_op')[0]
            self.loss=tf.get_collection('loss')[0]
            #restores weights etc.
            saver.restore(sess,full_model_name)
            Nin=input_data.shape[0]
            Nt_per_batch=(self.config.Nsteps_in*self.config.Nbatch)
            if (Nin < Nt_per_batch):
                print('Number of inputs < Number of batch expected')
                print('Padding with zeros')
                input_dat=np.append(input_dat,
                                    np.zeros((self.config.Nbatch-Nin,
                                              self.config.Noutputs)))
            nn_pred_total=np.zeros((Nin,self.config.Noutputs))
            i0=0
            end_of_file=False
            Niter=0
            #keep going over all data.
            while (i0 < Nin):
                Niter+=1
                #find size of remaining batch.
                Nb = min(self.config.Nbatch, int((Nin-i0)/self.config.Nsteps_in))
                print(i0,Nb*self.config.Nsteps_out)
                if (Nb*self.config.Nsteps_out<1):
                    print('No entries left.  Breaking loop')
                    break
                #now treat each time, as another element in a batch.
                #(i.e. march through dataset predicting, instead of randomly selecting for training)
                X_batch=np.zeros((Nb,self.config.Nsteps_in,self.config.Ninputs))

                #initialize some variables to get started.
                j0=i0;  j1=i0;
                for i in range(Nb):
                    j0=j1; j1=j0+self.config.Nsteps_out
                    X_batch[i,:,:]=input_data[j0:j1,:]
                    #step forward by Nsteps_out to forecast next period.
                    
                nn_pred=self.predict_on_batch(sess,X_batch)
                #now load into output.
                j1=i0; j0=i0;
                for i in range(Nb):
                    #step forward by Nsteps_out to forecast next period.
                    j0=j1;                     
                    j1=j0+self.config.Nsteps_out                
                    
                    if (j1<Nin):
                        nn_pred_total[j0:j1,:]=nn_pred[i]
                    else:
                        print('Hit End!')
                        j1=Nin-1
                        nn_pred_total[j0:j1,:]=nn_pred[i,j1-j0]
                        end_of_file=True
                        break
                i0=i0+self.config.Nsteps_out*Nb
            ## Original version    
            # i0=0
            # i1=self.config.Nbatch*self.
            # nn_pred_total=np.zeros((Nin,self.config.Noutputs))
            # while (i1 < Nin-self.config.Nsteps_in-self.config.Nsteps_out):
            #     print(i0,i1)
            #     #now treat each time, as another element in a batch.
            #     #(i.e. march through dataset predicting, instead of randomly selecting for training)
            #     X_batch=np.zeros((self.config.Nbatch,self.config.Nsteps_in,self.config.Ninputs))
                
            #     for i in range(self.config.Nbatch):
            #         j1 = j2
            #         j2 = j1+self.config.Nsteps_in
            #         X_batch[i,:,:]=input_data[(i0+i):(i0+i+self.config.Nsteps_in),:]
            #     nn_pred=self.predict_on_batch(sess,X_batch)
            #     sl=slice(self.config.Nsteps_in+i0,self.config.Nsteps_in+i1)
            #     nn_pred_total[sl]=nn_pred
            #     i0=i1
            #     i1+=self.config.Nbatch
            # #last iter: do remaining operations.  
            # Nleft=Nin-i0-self.config.Nsteps
            # X_batch=np.zeros((Nleft,self.config.Nsteps,self.config.Ninputs))
            # for i in range(Nleft):
            #     X_batch[i,:,:]=input_data[(i0+i):(i0+i+self.config.Nsteps),:]
            # nn_pred=self.predict_on_batch(sess,X_batch)
            # nn_pred_total[-Nleft:]=nn_pred
            # #nn_pred_reduced=np.round(nn_pred_total).astype(bool)
        return nn_pred_total

    
    def restore_model(self,sess,model_name,num=None):
        """Attempts to reset both TF graph, and 
        RNN stored variables/structure.
        """
        if (num==None):
            full_model_name=model_name+'-'+str(self.config.Nepoch)
        else:
            full_model_name=model_name+'-'+str(num)
        saver=tf.train.import_meta_graph(full_model_name+'.meta')
        #restore graph structure
        self.X=tf.get_collection('X')[0]
        self.y=tf.get_collection('y')[0]
        self.pred=tf.get_collection('pred')[0]
        self.train=tf.get_collection('train')[0]
        self.loss=tf.get_collection('loss')[0]
        #restores weights etc.
        saver.restore(sess,full_model_name)
    
class recurrent_NN(NNModel):
    """
    Make a multi-layer recurrent neural network for predicting next days
    stock data.
    """

    def make_RNN_cell(self,fn=tf.nn.relu):
        """
        Returns a new cell (for deep recurrent networks), with Nneurons,
        and activation function fn.

        Args: fn - tensorflow activation function, e.g. tf.nn.relu, tf.nn.tanh
        Return cell - TF RNN cell
        """
        #Make cell type
        if self.config.cell_type=='RNN':
            cell=BasicRNNCell(num_units=self.config.Nhidden,activation=fn)
        elif self.config.cell_type=='LSTM':
            cell=LSTMCell(num_units=self.config.Nhidden,activation=fn)
        elif self.config.cell_type=='GRU':
            cell=GRUCell(num_units=self.config.Nhidden,activation=fn)
        else:
            msg="cell_type must be RNN, LSTM or GRU. cell_type was {}".format(self.config.cell_type)
            raise Exception(msg)
        #only include dropout when training
        cell=DropoutWrapper(cell,input_keep_prob=self.keep_prob,
                            variational_recurrent=True,
                            input_size=self.config.Nhidden,
                            dtype=tf.float32)
        return cell
    
    def add_prediction_op(self):
        """The core model to the graph, that
        transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        """
        #Input matrix to change input dimensions to same size as hidden.
        A=tf.Variable(tf.random_uniform((self.config.Ninputs,self.config.Nhidden)),name="A")
        inputs_reduced=tf.tensordot(self.X,A,axes=[[2],[0]],name='inputs_reduced')
        #Make a cell for each layer 
        cell_list=[]
        for i in range(self.config.Nlayers):
            cell_list.append(self.make_RNN_cell(tf.nn.leaky_relu))
        multi_cell=tf.contrib.rnn.MultiRNNCell(cell_list,state_is_tuple=True)
        rnn_outputs,states=tf.nn.dynamic_rnn(multi_cell,inputs_reduced,dtype=tf.float32)

        #this maps the number of hidden units to fewer outputs.
        Nt2=self.config.Nhidden * self.config.Nsteps_in
        Nt1=self.config.Noutputs * self.config.Nsteps_out
        stacked_rnn_outputs = tf.reshape(rnn_outputs,[-1,Nt2])
        A_out=tf.Variable(tf.random_uniform((Nt2,Nt1)),name="A_out")
        remapped_outputs=tf.matmul(stacked_rnn_outputs,A_out)
        outputs=tf.reshape(remapped_outputs,[-1,self.config.Nsteps_out,self.config.Noutputs])
        
        return outputs

    

        
