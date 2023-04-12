import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer

import numpy as np

# from tensorflow.model import Model
import matplotlib.pyplot as plt

# to prevent creating huge logs.
from IPython.display import clear_output


class EBA_TF2_Base(object):
    """Tensorflow 2 Base Network class using the tf.keras api
    Aim is to take input data for past temperature data and energy usage,
    and return a prediction for future energy usage.
    """

    def __init__(self, Nstations=1):
        self.Nsteps = 24
        self.Ninputs = Nstations + 3
        self.Nneurons = 120
        self.Nlayers = 3
        self.Noutputs = 1  # number of stations to predict at that time.
        self.lr = 1e-2

        self.keep_prob = 0.9
        self.n_iter = 5000
        self.nout = 200
        self.build_network()

    def build_network(self):
        """Creates essential components for graph, and
        adds variables to instance.
        """
        tf.reset_default_graph()
        # Define inputs
        X = Input(tf.float32, [None, self.Nsteps, self.Ninputs], name="X")

        # Define functional model.
        # Could be LSTM with dense-layers.

        # Could be CNN with 1D convolutions.

        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """Adds placeholders to graph, by adding
        as instance variables for the model.
        """
        # load in the training examples, and their labels
        # inputs:  Nobs, with n_steps, and n_inputs per step
        X = tf.placeholder(tf.float32, [None, self.Nsteps, self.Ninputs], name="X")
        # Outputs: n_outputs we want to predict in the future.
        y = tf.placeholder(tf.float32, [None, self.Nsteps, self.Noutputs], name="y")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Make a feed_dict from inputs, labels as inputs for graph.
        Args:
        inputs_batch - batch of input data
        label_batch  - batch of output labels. (Can be none for prediction)
        Return:
        Feed_dict - the mapping from data to placeholders.
        """
        feed_dict = {self.X: inputs_batch}
        if labels_batch is not None:
            feed_dict[self.y] = labels_batch
        return feed_dict

    def make_RNN_cell(self, Nneurons, fn=tf.nn.relu):
        cell = tf.BasicRNNCell(num_units=Nneurons, activation=fn)
        return cell

    def add_prediction_op(self):
        """The core model to the graph, that
        transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        """

        # Make a list of cells to pass along.
        cell_list = []
        for i in range(self.Nlayers):
            cell_list.append(self.make_RNN_cell(self.Nneurons, tf.nn.relu))

        multi_cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
        # Note that using [cell]*n_layers did not work.  Might need to change init_state?
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
        # this maps the number of hidden units to fewer outputs.
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.Nneurons])
        stacked_outputs = fully_connected(
            stacked_rnn_outputs, n_outputs, activation_fn=None
        )
        outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

        return outputs

    def add_loss_op(self, outputs):
        """Add ops for loss to graph.
        Average mean-square error loss for a given set of outputs.
        Might need to upgrade to handle multiple outputs?
        """
        eps = 1e-15
        loss = tf.reduce_mean(tf.square(outputs - y))
        return loss

    def add_training_op(self, loss):
        """Create op for optimizing loss function.
        Can be passed to sess.run() to train the model.
        Return
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        training_op = optimizer.minimize(loss)
        return training_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch:  np.ndarray of shape (Nbatch, Nfeatures)
            labels_batch: np.ndarray of shape (Nbatch, 1)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (Nbatch, Nfeatures)
        Returns:
            predictions: np.ndarray of shape (Nbatch, 1)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def get_random_batch(self, X, y, n_batch, seq_len):
        """get_random_batch(Xsig,t,n_batch)
        Gets multiple random samples for the data.
        Samples generated by 'get_selection' function.
        Makes list of returned entries.
        Then combines together with 'stack' function at the end.

        X - matrix of inputs, (Nt, Ninputs)
        y - vector of desired outputs (Nt)
        n_batch - number of batches
        seq_len - length of sequence to extract in each batch

        Outputs:
        X_batch - random subset of inputs shape (Nbatch,seq_len,Ninputs)
        y_batch - corresponding subset of outputs (Nbatch,seq_len)
        """
        Nt, Nin = X.shape
        x_list = []
        y_list = []
        for i in range(n_batch):
            n0 = int(np.random.random() * (Nt - seq_len - 1))
            x_sub = X[n0 : n0 + seq_len]
            y_sub = y[n0 : n0 + seq_len]
            x_list.append(x_sub)
            y_list.append(y_sub)
        x_batch = np.stack(x_list, axis=0)
        y_batch = np.stack(y_list, axis=0)
        y_batch = y_batch.reshape([n_batch, seq_len, -1])
        return x_batch, y_batch

    def run_graph(self, temperature, demand, save_name):
        """run_graph

        Trains the deep recurrent network on demand and temperature data.

        """
        init = tf.global_variables_initializer()

        # save model and graph
        saver = tf.train.Saver()

        loss_tot = np.zeros(int(self.n_iter / self.nout + 1))
        plt.figure()
        with tf.Session() as sess:
            init.run()
            for iteration in range(self.n_iter + 1):
                # Get a random batch of training data.
                X_batch, y_batch = get_random_batch(
                    temperature, demand, self.Nbatch, self.Nsteps
                )

                current_loss = self.train_on_batch(sess, X_batch, y_batch)
                if (iteration) % self.nout == 0:
                    clear_output(wait=True)
                    current_pred = self.predict_on_batch(sess, X_batch)
                    print(
                        "iter #{}. Current log-loss:{}".format(iteration, current_loss)
                    )
                    print("\n")
                    # save the weights
                    saver.save(sess, save_name, global_step=iteration)
                    loss_tot[int(iteration / self.nout)] = current_loss
            plt.plot(loss_tot)
            plt.ylabel("Log-loss")
            plt.xlabel("Iterations x100")
            plt.show()

    def predict_all(self, model_name, Xin):
        """predict_all
        Load a saved Neural network, and predict the output labels
        based on input_data

        Input: model_name - string name to where model/variables are saved.
        input_data - transformed data of shape (Nobs,Nfeature).

        Output nn_pred_reduced - vector of predicted labels.
        """
        # tf.reset_default_graph()
        with tf.Session() as sess:
            Nt, Nin = Xin.shape
            nmax = int(Nt / n_steps)
            ytot = np.zeros((Nt, 1))

            loader = tf.train.import_meta_graph(model_name + ".meta")
            loader.restore(sess, model_name)
            i0 = 0
            i1 = self.Nbatch
            # restore variables
            # saver.restore(sess,model_path)
            for i in range(nmax - 1):
                n0 = n_steps * i
                x_sub = Xin[n0 : n0 + n_steps, :]
                x_sub = x_sub.reshape(-1, n_steps, Nin)
                y_pred = sess.run(outputs, feed_dict={X: x_sub})
                y_pred = self.predict_on_batch(sess, x_sub)
                ytot[n0 : n0 + n_steps] = y_pred

        return ytot

    def plot_whole_sample_fit(self, X, y, ntest, n_steps, model_name="pdx_RNN_model"):
        """plot_whole_sample_fit

        Plot ALL of the predictions of the trained model
        on a 'test' set with different noise, and longer
        times.  Concatenates the predicted results together.
        """
        # pull in the inputs, and predictions
        Nt, Nin = X.shape
        ytot = model_predict_whole(X, path_str)
        plt.figure()
        # now plot against the test sets defined earlier
        plt.plot(np.arange(0, ntest), X[:ntest, 0], "b", label="Training")
        plt.plot(np.arange(ntest, Nt), X[ntest:, 0], "g", label="Test")
        plt.plot(np.arange(Nt), ytot, "r", label="Predicted")
        plt.plot(np.arange(Nt), dem_mat, label="Real")
        plt.legend(loc="right")
        plt.show()
        return ytot
