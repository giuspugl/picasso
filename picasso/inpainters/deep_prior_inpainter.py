#
#
#
#
#   date: 2019-08-20
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019   Giuseppe Puglisi    gpuglisi@stanford.edu
#



import tensorflow as tf
from keras import layers

import keras.backend as K
from keras.layers import (
        Input,Conv2D,UpSampling2D,
        Dense ,Flatten,Reshape, ZeroPadding2D ,
        LeakyReLU,Masking
)
import numpy as np
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import RMSprop, Adam

from utils import MinMaxRescale

class   DeepPrior ():

    def upsampling_block(self, X, f, filters, stage, block ):

        conv_id =   block + str(stage)
        X = Conv2D( filters  ,  (f,f)  , padding='same',
               kernel_initializer = glorot_uniform(seed=0), name='conv1_'+conv_id)(X)
        X=LeakyReLU(alpha=.1, name='LeakyReLU1_'+conv_id)(X)
        X = Conv2D( filters  ,  (f,f)  , padding='same',
               kernel_initializer = glorot_uniform(seed=0), name='conv2_'+conv_id)(X)
        X=LeakyReLU(alpha=.1, name='LeakyReLU2_'+conv_id)(X)
        X = UpSampling2D((2,2),interpolation='nearest', name='upsample_'+conv_id)(X)

        return X

    def downsampling_block(self, X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- integer, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        conv_id =   block + str(stage)

        X = Conv2D( filters  ,  (f,f)  , padding='same',
               kernel_initializer = glorot_uniform(seed=0), name='conv1_'+conv_id)(X)
        X = Conv2D( filters  ,  (f,f) , strides=(s,s) , padding='valid', name='downsample_'+conv_id)(X)

        X=LeakyReLU(alpha=.1, name='LeakyReLU1_'+conv_id)(X)

        X = Conv2D( filters  ,  (f,f)  , padding='same',
               kernel_initializer = glorot_uniform(seed=0), name='conv2_'+conv_id)(X )
        X=LeakyReLU(alpha=.1, name='LeakyReLU2_'+conv_id)(X)

        return X

    def myloss(self, y_true,y_pred ):
        Kmask = K.not_equal(y_true,0 )
        y1= tf.boolean_mask(y_true,Kmask)
        y2= tf.boolean_mask(y_pred,Kmask)
        #return K.sum(K.sqrt(K.square(   y2  -y1    )))  # l1
        return K.sqrt(K.sum(K.square(   y2  -y1    ), axis=-1)) /K.sqrt(K.sum(K.square(   y1 ), axis=-1))
    def my_accuracy (self,  y_true ,y_pred ):
        Kmask = K.not_equal(y_true,0 )
        y1= tf.boolean_mask(y_true,Kmask)
        y2= tf.boolean_mask(y_pred,Kmask)
        return   K.sqrt(K.sum(K.square(   y2   ), axis=-1))  /K.sqrt(K.sum(K.square(   y1 ), axis=-1))

    def __init__(self,
        input_shape,
        nd =[16,32,64,128  ,128  ,128],
        verbose = False  ):
        self.rdseed=123456789
        self.verbose=verbose
        X_input =  Input(   input_shape )
        nu=nd[::-1]
        kd =[3] * len(nu)
        ku = [5]*len(nu)
        X= self.downsampling_block(X_input, kd[0], nd[0], 0, 'd')
        #encoder block
        for  i in  range(1, len(nu)):
            X= self.downsampling_block(X , kd[i], nd[i], i, 'd')
        for  i in  range( len(nu)):
            X= self.upsampling_block(X , ku[i], nu[i], i, 'u')

        X = UpSampling2D((2,2),interpolation='nearest', name='upsample_last')(X)
        #encoder
        #X = ZeroPadding2D((2, 2))(X_input)
        #X= Dense( 1 ,activation='sigmoid', name='FC1')(X)
        X = Conv2D( 1  ,  (5,5) , padding='same',
                   kernel_initializer = glorot_uniform(seed=0), name='lastconv')(X)
        X=LeakyReLU(alpha=.1, name='LeakyReLU_last' )(X)

        #X= Dense( 1 ,activation='relu', name='FC')(X)

        self.model = Model(inputs = X_input, outputs = X , name='DeepPrior')


    def summary (self) :
        self.model.summary()
    def compile (self, optimizer   ) :
        self.model.compile(loss=self.myloss, optimizer=optimizer, metrics=[self.my_accuracy] )

    def train(self, z , X ,epochs =2000, verbose=False ) :
        train_out = self.model.fit(z, X, epochs=epochs , verbose=verbose )
        self.loss=train_out.history['loss']
        self.accuracy =  train_out.history['my_accuracy']
    def evaluate (self, z, X) :
        self.preds =  self.model.evaluate(x=z, y=X)
        if self.verbose :
            print ("Loss = " + str(self.preds[0]))
            print ("Test Accuracy = " + str(self.preds[1]))

    def predict(self,) :
        pred= self.model.predict(self.Z)
        return pred

    def setup_input(self,fname_masked    ):
        maskdmap=np.load(fname_masked)
        holemask = np.ma.masked_not_equal(maskdmap,0  ) .mask
        a=0; b=1
        maxval = maskdmap[holemask].max() ; minval = maskdmap[holemask].min()
        maskdmap =MinMaxRescale(maskdmap,a =a ,b=b )
        #maskdmap = (maskdmap -minval) / (maxval - minval) #rescale to  (0,1)

        maskdmap[np.logical_not( holemask)]=0.

        maskdmap = np.expand_dims(np.expand_dims( maskdmap, axis=0), axis=-1)

        randstate= np.random.RandomState(self.rdseed)
        noisemap =     randstate.uniform(low=a, high=b/10 ,  size=maskdmap.shape )  # see the Deep prior paper for the choice b=1./10 
        self.X = maskdmap; self.Z = noisemap ;
        self.min = minval;  self.max = maxval

        pass

    def rescale_back (self, v ) :
        return MinMaxRescale(v, a= self.min , b = self.max )
        #return  ( v* (self.max - self.min) +
        #            self.min )
