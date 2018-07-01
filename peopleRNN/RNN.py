
# coding: utf-8

# In[ ]:

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import shutil
import sys

import tools
from scipy import linalg
import numpy as np
# In[ ]:

class RNNmodel:
    '''

    '''
    def __init__(self, config):
        self.config = config
        self.ID = config['ID']
        self.rnn_ID = config['rnn_ID']
        self.n_iter = config['n_iter']
        self.n_prt = config['n_prt']
        self.n_input = config['n_input']
        self.n_joint = config['n_joint']
        self.n_step = int((config['n_step']))
        self.n_output = config['n_output']
        self.n_batch = config['n_batch']
        self.n_save = config['n_save']
        self.n_history = config['n_history']
        self.LR = config['LR']
        
        self.history = {
            'train' : [],
            'test' : []
        }
        self.checkpoint = 0
        self.path = './{0}/{1}'.format(self.ID, self.rnn_ID)
        try: 
            os.mkdir(self.path)
            os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
        except FileExistsError:
            msg = input('[FileExistsError] Will you remove directory? [Y/N] ')
            if msg == 'Y': 
                shutil.rmtree(self.path)
                os.mkdir(self.path)
                os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
            else: 
                print('Please choose another ID')
                assert 0
        # tensorflow 작동을 위한 그래프 생성                  
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 데이터가 들어갈 껍데기인 placeholder 생성
            self.x = tf.placeholder(tf.float32, [None, self.n_step, self.n_input], name='x')
            # 라벨의 형태
            self.y = tf.placeholder(tf.int32, [None], name='y')
            # rnn 모델의 길이를 저장할 placeholder
            self.seq_len = tf.placeholder(tf.int32, [None])
            
            # 각 라벨별 rnn 모델 생성
            each_loss = []
            
            # 각각의 라벨에 모델이생김(그 라벨만 쳐다보는 모델, 즉, 그 라벨이거나 아니거나 라는 문제를 품)
            with tf.variable_scope("model_label"):
                self.pred = self.clf(self.x)
                each_loss.append(self.compute_loss(self.pred, tf.one_hot(self.y, self.n_output)))
            
            # Optimize                           
            self.loss = tf.reduce_sum(each_loss)
            self.LR_decay = tf.train.exponential_decay(self.LR, tf.Variable(0, trainable=False), 500, 0.9, staircase=True)
            self.optm = tf.train.AdamOptimizer(learning_rate=self.LR_decay).minimize(self.loss)
            
            # 학습 될 weight의 값을 초기화
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.init)
        
        print('Model ID : {}'.format(self.ID))
        print('Model saved at : {}'.format(self.path))


    ## Classifier
    def clf(self, x):
        with tf.variable_scope('clf'):
            h = self.rnn_layer(x, 150, 'birnn')
            pred = self.rnn_fc_layer(h, 'pred', 
                                      n_in = 150,
                                      n_out = self.n_output)
        return pred
        ## Compute loss
    def compute_loss(self, pred, y):
        with tf.variable_scope('compute_loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=pred, labels=y))
          
        return loss
     ## Layers  : 네트워크를 구성하는 레이어들
    def rnn_layer(self, x, n_hidden, name):
        '''
        <input>
        shape : (batchsize , n_step, n_input)
        '''
        with tf.variable_scope(name):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            output_h, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
                                                          sequence_length=self.seq_len)
            return output_h

    def rnn_fc_layer(self, input_tensor, name, n_in, n_out):
        forward_h = tf.unstack(input_tensor, axis=1)
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
            bias = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            logit = tf.matmul(forward_h[-1], weights) + bias
            #logit = tf.transpose(logit, [1, 0, 2])
            #return tf.reduce_sum(logit, 1)
            return logit
     

    
    ## Train
    def fit(self, data, label):
        self.data = MOTION(data, label, self.config)
        for epoch in range(1, self.n_iter+1):
            train_x, train_y  = self.data.train.next_batch(self.n_batch)
            train_seq_len = self.length(train_x)
            self.sess.run(self.optm, feed_dict={self.x : train_x, self.y : train_y, self.seq_len : train_seq_len})
            
            if epoch % self.n_prt == 0:
                train_loss = self.get_loss(train_x, train_y)
                print('Your loss ({0}/{1}) : {2}'.format(epoch, self.n_iter, train_loss))
                
            if epoch % self.n_save == 0:
                self.checkpoint += self.n_save
                self.save('{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.rnn_ID, self.checkpoint))
            
            if epoch % self.n_history == 0:
                test_x, test_y = self.data.test.next_batch(self.n_batch)
                train_loss = self.get_loss(train_x, train_y)
                test_loss = self.get_loss(test_x, test_y)
                self.history['train'].append(train_loss)
                self.history['test'].append(test_loss)
            sys.stdout.write('\r'+str(epoch)+'/'+str(self.n_iter+1))
       
    
    ## Analysis : 학습이후 분석을위한 함수들
    def predict(self, x):
        result = self.sess.run(self.pred, feed_dict={self.x : x, self.seq_len : self.length(x)})
        #pred = np.argmax(pred, axis=1)
        return result
    
    def get_loss(self, x, y):
        loss = self.sess.run(self.loss, feed_dict={self.x : x, self.y : y, self.seq_len : self.length(x)})
        return loss
    
    def length(self, sequences):
        return np.asarray([len(s) for s in sequences], dtype=np.int64)
    
    
            
    ## Save/Restore
    def save(self, path):
        self.saver.save(self.sess, path)
        
    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))
        
## batch maker    
# batch에 쓰일 데이터셋은 SlideByWindow함수를 이용
'''
train_x, train_y  = self.data.train.next_batch(self.n_batch)
이런식으로 쓰임
'''
class BatchMaker:
    def __init__(self, data, label, config, train=True):
        self.n_class = config['n_output']
        self.n_step = config['n_step']
        self.n_input = config['n_input']
        self.hop = config['hop']
        self.n_joint = config['n_joint']
        self.train = train
        
        data_length = len(data)
        if self.train==True:
            self.E_data, self.E_label = data, label
                             
        else:
            self.E_data, self.E_label = data, label

    def next_batch(self, n_batch):
        rand_idx = np.random.randint(0, self.E_data.shape[0], n_batch)

        batch = self.E_data[rand_idx]
        label = self.E_label[rand_idx]
        if self.train==True:
            batch_aug = batch.copy()
            rand_aug = 0.5 + 1.5*np.random.rand(batch.shape[0]) 
            batch_aug[:, :, [3*(i)+1 for i in range(20)]] = batch_aug[:, :, [3*(i)+1 for i in range(20)]]*rand_aug.reshape(-1, 1, 1)
            return batch_aug, label
        else:
            
            return batch, label

class MOTION:
    def __init__(self, data, label, config):

        self.train = BatchMaker(data, label, config) 
        self.test = BatchMaker(data, label, config, train=False)
