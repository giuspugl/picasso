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

from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import RMSprop, Adam
import numpy as np


from utils import MinMaxRescale

class   DeepPrior ():
    """
    Inpainting class interface for Deep-Prior
    """

    def upsampling_block(self, X, f, filters, stage, block ):
        """
        Implementation of the up-sampling  block

        **Parameters**

        - ``X``:
            input tensor of shape ``(m, n_H_prev, n_W_prev, n_C_prev)``

        - ``f``: {int}
            specifying the shape of the middle CONV's window for the main path
        - ``filters`` :{int}
            defining the number of filters in the CONV layers of the main path
        - ``stage`` :{int}
            used to name the layers, depending on their position in the network
        - ``block``: {string}
            used to name the layers, depending on their position in the network


        **Returns**

        - ``X``:
            output of the convolutional block, tensor of shape ``(n_H, n_W, n_C)``

        """

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
        Implementation of a down-sampling block

        **Parameters**

        - ``X``:
            input tensor of shape ``(m, n_H_prev, n_W_prev, n_C_prev)``

        - ``f``: {int}
            specifying the shape of the middle CONV's window for the main path
        - ``filters`` :{int}
            defining the number of filters in the CONV layers of the main path
        - ``stage`` :{int}
            used to name the layers, depending on their position in the network
        - ``block``: {string}
            used to name the layers, depending on their position in the network


        **Returns**

        - ``X``:
            output of the convolutional block, tensor of shape ``(n_H, n_W, n_C)``

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
        """
        Loss function evaluated outside the masked region and normalized by  the ground-truth norm.

        """
        Kmask = K.not_equal(y_true,0 )
        y1= tf.boolean_mask(y_true,Kmask)
        y2= tf.boolean_mask(y_pred,Kmask)
        return K.sqrt(K.sum(K.square(   y2  -y1    ), axis=-1)) /K.sqrt(K.sum(K.square(   y1 ), axis=-1))

    def my_accuracy (self,  y_true ,y_pred ):
        """
        Accuracy defined as the ratio between  the 2-norms of the prediction and of the ground-truth.
        """
        Kmask = K.not_equal(y_true,0 )
        y1= tf.boolean_mask(y_true,Kmask)
        y2= tf.boolean_mask(y_pred,Kmask)
        return   K.sqrt(K.sum(K.square(   y2   ), axis=-1))  /K.sqrt(K.sum(K.square(   y1 ), axis=-1))

    def __init__(self,
        input_shape,
        nd =[16,32,64,128  ,128  ,128],
        verbose = False, meshgrid=True   ):


        """
        Initialize the Deep -Prior network with parameters reported in the prescription of
         `Deep-Prior Supplement Material <https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM#pdfviewer>`_
        """

        self.rdseed=123456789
        self.verbose=verbose
        self.meshgrid=meshgrid
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
        X = Conv2D( 1  ,  (5,5) , padding='same',
                   kernel_initializer = glorot_uniform(seed=0), name='lastconv')(X)
        X=LeakyReLU(alpha=.1, name='LeakyReLU_last' )(X)

        #X= Dense( 1 ,activation='relu', name='FC')(X)

        self.model = Model(inputs = X_input, outputs = X , name='DeepPrior')


    def summary (self) :
        """
        Returns the model summary

        """
        self.model.summary()
    def compile (self, optimizer   ) :
        """
        Compile the architecture
        """
        self.model.compile(loss=self.myloss, optimizer=optimizer, metrics=[self.my_accuracy] )

    def train(self, z , X ,epochs =2000, verbose=False  ) :
        """
        Iterate over the epochs
        """
        train_out = self.model.fit(z, X, epochs=epochs , verbose=verbose )
        self.loss=train_out.history['loss']
        self.accuracy =  train_out.history['my_accuracy']
    def evaluate (self, z, X) :

        self.preds =  self.model.evaluate(x=z, y=X)
        if self.verbose :
            print ("Loss = " + str(self.preds[0]))
            print ("Test Accuracy = " + str(self.preds[1]))

    def predict(self,) :
        """
        Inpainting with Deep -Prior
        """
        pred= self.model.predict(self.Z)
        pred = self.X *np.int_ (self.mask) + pred * (1-np.int_( self.mask ) )

        return pred

    def setup_input(self,fname_masked    ):
        """
        Preprocessing the corrupted image and setting up the generative images.
        By default deep-prior will generate pixel values starting from an image with
        uniformly random  distributed pixels (in :math:`[0,1/10]`).
        Otherwise if ``meshgrid==True``, generator is a set of 4 images, i.e.  a upward, downward, leftward and rightward diagonal
        gradients_summary.
        This mitigates the gridding pattern injected in the reconstruction area.
        """
        maskdmap=np.load(fname_masked)
        if self.meshgrid :
            x = np.linspace(0, 1, maskdmap.shape[0])
            y = np.linspace(0, 1, maskdmap.shape[1])
            xv, yv = np.meshgrid(x, y)
            up =  (xv-yv  +1)/2
            down =  (xv+yv  )/2

        holemask = np.ma.masked_not_equal(maskdmap,0  ) .mask

        a=0; b=1
        maxval = maskdmap[holemask].max() ; minval = maskdmap[holemask].min()
        maskdmap =MinMaxRescale(maskdmap,a =a ,b=b )

        maskdmap[np.logical_not( holemask)]=0.

        maskdmap = np.expand_dims(np.expand_dims( maskdmap, axis=0), axis=-1)
        self.mask =np.expand_dims(np.expand_dims( holemask, axis=0), axis=-1)
        randstate= np.random.RandomState(self.rdseed)
        noisemap =     randstate.uniform(low=a, high=b ,  size=maskdmap.shape )  # see the Deep prior paper for the choice b=1./10

        self.min = minval;  self.max = maxval
        if self.meshgrid :
            self.Z =np.expand_dims(np.array([xv,yv, up,down] ).T,0)
            self.X = maskdmap;
        else:
            randstate= np.random.RandomState(self.rdseed)
            noisemap =     randstate.uniform(low=a, high=b/10  ,  size=maskdmap.shape )
            self.X = maskdmap;
            self.Z = noisemap ;
        pass

    def rescale_back (self, v ) :
        """
        rescale inpainted map to physical units .
        """
        return MinMaxRescale(v, a= self.min , b = self.max )
