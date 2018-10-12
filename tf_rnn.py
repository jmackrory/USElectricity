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
    def __init__(self,Nsteps_in=48, Nsteps_out=24,
                 Ninputs=1, Noutputs=1,
                 cell='RNN',Nlayers=2,Nhidden=100,
                 lr=0.001,Nepoch=1000,keep_prob=0.5,
                 Nprint=20,Nbatch=100):
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
        self.keep_prob=keep_prob
        self.Nepoch=Nepoch
        self.Nprint=Nprint
        #only grabbing a fraction of the data
        self.Nbatch=Nbatch

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
            m0 = n1-self.config.Nsteps_out
            m1 = n1
            x_sub = X[n0:n1]
            y_sub = y[m0:m1]
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

                    nn_pred=self.predict_on_batch(sess,X_batch[:2])
                    self.plot_data(X_batch[0],y_batch[0],nn_pred[0])
                    #save the weights
                    if (save_name != None):
                        saver.save(sess,save_name,global_step=iteration,
                                   write_meta_graph=True)
                    #manual logging of loss    
                    loss_tot[int(iteration/self.config.Nprint)]=current_loss
            writer.close()
            #Manual plotting of loss.  Writer/Tensorboard supercedes this .
            self.plot_losses(loss_tot)

    def plot_losses(self,loss_tot):
        """plot_losses
        Make linear and log-plots of losses for the latest batch.
        """
        plt.figure()
        plt.subplot(121)
        plt.plot(loss_tot)
        plt.ylabel('Error')
        plt.xlabel('Iterations x{}'.format(self.config.Nprint))
        plt.subplot(122)
        plt.semilogy(loss_tot)
        plt.ylabel('Error')
        plt.xlabel('Iterations x{}'.format(self.config.Nprint))
        plt.show()
        return None
        
    def plot_data(self,X,y,pred):
        plt.figure()
        t1=np.arange(self.config.Nsteps_in)
        t2=self.config.Nsteps_in+np.arange(self.config.Nsteps_out)
        plt.plot(t1,X)
        plt.plot(t2,y,label='actual')
        plt.plot(t2,pred,label='predicted')
        plt.legend()
        plt.show()
        return None
            
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
                j0=i0;  j1=i0; j2=i0;
                for i in range(Nb):
                    j0=j2;
                    j1=j0+self.config.Nsteps_in
                    j2=j0+self.config.Nsteps_out
                    X_batch[i,:,:]=input_data[j0:j1,:]
                    #step forward by Nsteps_out to forecast next period.
                    
                nn_pred=self.predict_on_batch(sess,X_batch)
                #now load into output.
                j2=i0; j0=i0;
                for i in range(Nb):
                    #step forward by Nsteps_out to forecast next period.
                    # j0=j1;                     
                    # j1=j0+self.config.Nsteps_out                
                    j0=j2;
                    #j1=j0+self.config.Nsteps_in
                    j2=j0+self.config.Nsteps_out                    
                    if (j2<Nin):
                        nn_pred_total[j0:j2,:]=nn_pred[i]
                    else:
                        print('Hit End!')
                        j2=Nin-1
                        nn_pred_total[j0:j2,:]=nn_pred[i,j2-j0]
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
    
class recurrentNeuralNetwork(NNModel):
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
        #always include dropout when training, but tweak keep_prob to turn this off.
        cell=DropoutWrapper(cell,input_keep_prob=self.keep_prob,
                            variational_recurrent=True,
                            input_size=self.config.Nhidden,
                            dtype=tf.float32)
        return cell
    
    def add_prediction_op(self):
        """add_prediction_op

        The core model to the graph, that transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        
        Maps from Ninputs to Nhidden dimensions for each input timestep.
        Then has Nlayer RNN with dropout.
        Then maps outputs from Nhidden to Noutput dimensions.
        """
        #Input matrix to change input dimensions to same size as hidden.
        A=tf.Variable(tf.random_uniform((self.config.Ninputs,self.config.Nhidden)),name="A",trainable=True)
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
        A_out=tf.Variable(tf.random_uniform((Nt2,Nt1)),name="A_out",trainable=True)
        remapped_outputs=tf.matmul(stacked_rnn_outputs,A_out)
        outputs=tf.reshape(remapped_outputs,[-1,self.config.Nsteps_out,self.config.Noutputs])
        
        return outputs

    
       
class simpleRecurrentNeuralNetwork(NNModel):
    """Simple original RNN Model

    Simple network, currently preserved in EBA_RNN_Old.
    Only designed to work for a single output.

    """
    def add_prediction_op(self):
        #Make a list of cells to pass along.  
        cell_list=[]

        cell_list=[BasicRNNCell(num_units=self.config.Nhidden,activation=tf.nn.relu)
                   for i in range(self.config.Nlayers)]
        # for i in range(self.config.Nlayers):
        #     cell_list.append(make_RNN_cell(self.config.Nhidden,tf.nn.relu))

        multi_cell=tf.contrib.rnn.MultiRNNCell(cell_list,state_is_tuple=True)
        rnn_outputs,states=tf.nn.dynamic_rnn(multi_cell,self.X,dtype=tf.float32)
        #this maps the number of hidden units to fewer outputs.
        stacked_rnn_outputs = tf.reshape(rnn_outputs,[-1,self.config.Nhidden])
        stacked_outputs = fully_connected(stacked_rnn_outputs,self.config.Noutputs,activation_fn=None)
        outputs=tf.reshape(stacked_outputs,[-1,self.config.Nsteps_out,self.config.Noutputs])
        return outputs


    # def make_RNN_cell(self,fn=tf.nn.relu):
    #     cell=BasicRNNCell(num_units=self.n_neurons,activation=fn)
    #     return cell
    
    def get_random_batch(self,X,y):
        """get_random_batch(Xsig,t,n_batch)   
        Gets multiple random samples for the data.
        Samples generated by 'get_selection' function.
        Makes list of returned entries.
        Then combines together with 'stack' function at the end.

        X - matrix of inputs, (Nt, Ninputs)
        y - vector of desired outputs (Nt)

        Outputs:
        X_batch - random subset of inputs shape (config.Nbatch,config.Ntimes_out,config.Ninputs) 
        y_batch - corresponding subset of outputs (config.Nbatch,config.Ntimes_out)
        """
        Nt,Nin = X.shape
        x_list=[]
        y_list=[]
        for i in range(self.config.Nbatch):
            n0=int(np.random.random()*(Nt-self.config.Nsteps_out-1))
            n1 = n0+self.config.Nsteps_out
            x_sub = X[n0:n1]
            y_sub = y[n0:n1]
            x_list.append(x_sub)
            y_list.append(y_sub)
        x_batch=np.stack(x_list,axis=0)
        y_batch=np.stack(y_list,axis=0)
        y_batch=y_batch.reshape( [self.config.Nbatch,self.config.Nsteps_out,-1])                    
        return x_batch,y_batch

    
    def predict_all(self,Xin,path_str="pdx_RNN_model"):
        """model_predict_whole(tstart)
        Retrieve the outputs of the network for all values of the inputs 
        """
        Nt,Nin=Xin.shape
        nmax = int(Nt/n_steps)
        ytot = np.zeros((Nt,1))
        #Note that loading/saving graph is not properly implemented yet.    
        #reset graph, and reload saved graph
        tf.reset_default_graph()
        model_path = "./models/"+path_str    
        saver = tf.train.import_meta_graph(model_path+".meta")
        #restore graph structure
        X=tf.get_collection('X')[0]
        y=tf.get_collection('y')[0]
        outputs=tf.get_collection('pred')[0]
        train_op=tf.get_collection('train_op')[0]
        loss=tf.get_collection('loss')[0]
        #restores weights etc.
        #saver.restore(sess,full_model_name)

        with tf.Session() as sess:

            #restore variables
            saver.restore(sess,model_path)
            for i in range(nmax-1):
                n0=n_steps*i
                n1 = n0+self.config.Nsteps_out
                x_sub = Xin[n0:n1,:]
                x_sub = x_sub.reshape(-1,self.config.Nsteps_out,Nin)
                y_pred=sess.run(outputs,feed_dict={X:x_sub})
                #nn_pred=predict_on_batch(sess,X_batch)            
                ytot[n0:n1]=y_pred
        return ytot
