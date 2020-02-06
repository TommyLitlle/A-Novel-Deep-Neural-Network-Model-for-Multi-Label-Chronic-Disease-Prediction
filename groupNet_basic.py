from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sys
import numpy as np
import collections
from sklearn import preprocessing
import time
from tensorflow.python.ops import array_ops

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

def input_fn(data_set):
    # convert csv file to matrix
    data_matrix = np.loadtxt(open(data_set, "rb"), delimiter=",", skiprows=0)
    np.random.shuffle(data_matrix)
    data = data_matrix[:, :62]
    label_1= data_matrix[:, 62:63]
    length =len(data)
    
    perm=np.arange(length)
    np.random.shuffle(perm)
    print(perm)
    data=data[perm]
    label_1=label_1[perm]
    
    #one_hot1
    enc = preprocessing.OneHotEncoder() 
    enc.fit(label_1)  
    label_1 = enc.transform(label_1).toarray() 
    
    #concatenation
    label=np.reshape(label_1,(length,8))
    #convert to array 
    data = np.asarray(data, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    train_data = data[:88000]
    eval_data = data[88000:]

    train_labels = label[:88000]

    eval_labels = label[88000:]


    return train_data, train_labels, eval_data, eval_labels



    
class DataSet(object):

    def __init__(self,
                 data,
                 labels,
                 one_hot=False,
                 dtype=np.float32,
                 reshape=True):
            
           
        self._num_examples=data.shape[0]
            #print(self._num_examples)
        if reshape:
            #assert data.shape[2] ==1
            data = data.reshape (data.shape[0],
                                      data.shape[1])
        if dtype == np.float32:
            data = data.astype(np.float32)
        self._data = data
        self._labels= labels
        self._epochs_completed =0
        self._index_in_epoch =0
        
    @property
    def data(self):
        return self._data
            
    @property
    def labels(self):
        return self._labels
            
    @property
    def num_exanples(self):
        return self._num_examples
            
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
        
    def next_batch(self, batch_size, shuffle =True):
        """ Return the bext 'batch_size' examples from this data set."""
        start=self._index_in_epoch
                
        #Shuffle for the first epoch
                
        if self._epochs_completed ==0 and start ==0 and shuffle:
            perm0=np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data=self.data[perm0]
            self._labels=self.labels[perm0]
                    
        #go to the next epoch
        if start+batch_size > self._num_examples:
            #finished epoch
            self._epochs_completed +=1
            #get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start : self._num_examples]
            labels_rest_part =self._labels[start: self._num_examples]
            #Shuffle the data      
            if shuffle:
                perm=np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data=self.data[perm]
                self._labels=self.labels[perm]
                        
            # Start next epoch
                start = 0
                self._index_in_epoch =batch_size-rest_num_examples
                end=self._index_in_epoch
                data_new_part = self._data[start: end]
                labels_new_part = self._labels[start: end]
                return np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)                    
        else:
            self._index_in_epoch += batch_size
            end=self._index_in_epoch
            return self._data[start: end], self._labels[start: end]

          
                    
def read_data_sets( fake_data=False,
                    one_hot =False,
                    dtype = np.float32,
                    reshape = True,):
                     
    train_data , train_labels,test_data, test_labels=input_fn('table_11.csv')
        
    
    train =DataSet(train_data, train_labels, dtype =dtype, reshape =reshape)
    test = DataSet(test_data, test_labels, dtype=dtype, reshape =reshape)
    
    return Datasets(train=train, test=test)

#module

def Inception(inputs,num_filters,activation,alpha=1):
        
        conv1=tf.contrib.layers.conv2d(inputs,
                                        num_filters,
                                        kernel_size = [1,3],
                                        padding='same')
        norm = tf.layers.batch_normalization(conv1)
        conv1=activation(norm)
        
        conv1=tf.contrib.layers.conv2d(conv1,
                                        num_filters,
                                        kernel_size = [1,3],
                                        padding='same')
        norm = tf.layers.batch_normalization(conv1)
        conv1=activation(norm)
        
        conv2 = tf.contrib.layers.conv2d(inputs,
                                  num_filters*alpha,
                                  kernel_size = [1,3],
                                  padding='same')
        norm = tf.layers.batch_normalization(conv2)
        conv2=activation(norm)
        
        conv2 = tf.contrib.layers.conv2d(conv2,
                                  num_filters*alpha,
                                  kernel_size = [1,3],
                                  padding='same')
        norm = tf.layers.batch_normalization(conv2)
        conv2=activation(norm)
        
        conv =tf.concat([conv1,conv2],axis=3)
        
        return conv 



def Inception_2(inputs,num_filters,activation,alpha=1):
        
        conv1=tf.contrib.layers.conv2d(inputs,
                                        num_filters,
                                        kernel_size = [1,3],
                                        padding='valid')
        norm = tf.layers.batch_normalization(conv1)
        conv1=activation(norm)
        
        conv1=tf.contrib.layers.conv2d(conv1,
                                        num_filters,
                                        kernel_size = [1,2],
                                        padding='valid')
        norm = tf.layers.batch_normalization(conv1)
        conv1=activation(norm)
        
        conv2 = tf.contrib.layers.conv2d(inputs,
                                  num_filters*alpha,
                                  kernel_size = [1,2],
                                  padding='valid')
        norm = tf.layers.batch_normalization(conv2)
        conv2=activation(norm)
        
        conv2 =tf.contrib.layers.conv2d(conv2,
                                  num_filters*alpha,
                                  kernel_size = [1,3],
                                  padding='valid')
        norm = tf.layers.batch_normalization(conv2)
        conv2=activation(norm)
        
        conv =tf.concat([conv1,conv2],axis=3)
        
        return conv 

def separable_conv(x):
    
        conv = tf.layers.separable_conv2d(x, filters=1,kernel_size=[1,1],padding='valid')
        norm = tf.layers.batch_normalization(conv)
        conv=relu(norm)
   
        return conv


def relu(x, alpha=0.4, max_value=None):

    negative_part = tf.nn.relu6(-x)
    x = tf.nn.relu6(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(alpha, dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


def focal_loss(onehot_labels, cls_preds,
                            alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute sigmoid focal loss between logits and onehot labels
    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)
    Args:
      onehot_labels: Each row labels[i] must be a valid probability distribution
      cls_preds: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)
    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)

        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        predictions = tf.nn.sigmoid(precise_logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), precise_logits, 1.-precise_logits)
        # add small value to avoid 0
        epsilon = 1e-8
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                     name=name, axis=1)
        return losses

class CNN:
    def __init__(self,alpha,batch_size,num_classes,num_features):
        """ Initialize the CNN model
        :param alpha: the learning rate to be used by the model
        :param batch_size : the number of batches to use for training
        :param num_classes: the number of classes in the dataset
        :param num_features: the number of features in the dataset
        """
        
        self.alpha= alpha
        self.batch_size = batch_size
        self.name='CNN'
        self.num_classes = num_classes
        self.num_features = num_features
        
        def __graph__():
        
            #[batch_size, num_features]
            x_input = tf.placeholder(dtype=tf.float32,shape=[None, num_features], name='x_input')
            
            #[batch_size, num_classes*num_labels]
            y_input=tf.placeholder(dtype= tf.float32, shape=[None,num_classes], name='actual_label')
            
            
            input_layer = tf.reshape(x_input,[-1,1,62,1])
        
            conv=Inception(input_layer,12,relu)
            
            #conv. 1x1
            conv = tf.layers.conv2d(
                inputs=conv,
                filters=12,
                kernel_size=[1,1],
                padding="same")
            
            norm = tf.layers.batch_normalization(conv)
            activation =relu(norm)
            # Pooling Layer #1
            pool = tf.layers.max_pooling2d(inputs=activation, pool_size=[1, 2], strides=2)
            
            
            # Dropout, to avoid over-fitting
            keep_prob = tf.placeholder(tf.float32)
            dropout= tf.layers.dropout(pool,keep_prob)
            
            
            #flatten abstract feature
            flat_1 = tf.reshape(dropout,[-1, 12*31])
            flat_1 =tf.layers.dense(flat_1, units=372, activation= relu)
            
    
            #classification 
            digit1 = tf.layers.dense(flat_1, units=8)
            #loss function
            digit1_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_input, digit1))
            
            digit1 = tf.identity(tf.nn.softmax(digit1))
          
            loss = digit1_loss
            tf.summary.scalar('loss',loss)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
            
            #accuracy
          
            output= tf.argmax(digit1,1)
            label = tf.argmax(y_input,1)
            
            correct_pred= tf.equal(output, label)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            tf.summary.scalar('accuracy', accuracy)
            
            merged = tf.summary.merge_all()
            
            self.x_input = x_input
            self.y_input  = y_input
            self.keep_prob =keep_prob
            self.digit1=digit1
            self.loss = loss
            self.optimizer=optimizer
            self.accuracy = accuracy
            self.merged = merged
            
        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')
        

    def train(self,checkpoint_path, epochs,log_path, train_data,test_data):
        """Trains the initialized model.
        :param checkpoint_path: The path where to save the trained model.
        :param log_path: The path where to save the TensorBoard logs.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :return: None
        """
        
        if not os.path.exists(path=log_path):
            os.mkdir(log_path)
            
        if not os.path.exists(path=checkpoint_path):
            os.mkdir(checkpoint_path)
            
        
        saver= tf.train.Saver(max_to_keep=4)
        
        init = tf.global_variables_initializer()
        
        timestamp = str(time.asctime())
            
        train_writer = tf.summary.FileWriter(logdir=log_path +'-training', graph=tf.get_default_graph())
        
        with tf.Session() as sess:
            
            best_accuracy = 0.8003
            sess.run(init)
            
            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
            
            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                
            for index in range(epochs): 
                #train by batch
                batch_features, batch_labels = train_data.next_batch(self.batch_size)
                
                #input dictionary with dropout of 50%
                feed_dict = {self.x_input:batch_features, self.y_input:batch_labels}
                
                # run the train op
                summary, _, loss = sess.run([self.merged, self.optimizer, self.loss], feed_dict=feed_dict)
                
                feed_dict ={self.x_input:test_data.data, self.y_input:test_data.labels}
                # get the accuracy of training
                train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
                
                print('step: {}, training accuracy : {}, training loss : {}\n'.format(index, train_accuracy, loss))
                
                if train_accuracy >= best_accuracy:
                    best_accuracy = train_accuracy 
                    #dispaly the training accuracy
                    print('step: {}, best accuracy : {}'.format(index, best_accuracy))
                    saver.save(sess, save_path=os.path.join(checkpoint_path, self.name), global_step=index) 
         


if __name__ == '__main__':
    
    data=read_data_sets()
    train_data=data.train
    test_data = data.test
    num_classes=8      
    num_features = 62
    model=CNN(alpha=0.003, batch_size=128, num_classes=num_classes, num_features=num_features)
    model.train(checkpoint_path='C:/tmp/LP10convnet_model11',epochs=30000, log_path='C:/tmp/tensorflow/logs',
                train_data=train_data, test_data=test_data)
