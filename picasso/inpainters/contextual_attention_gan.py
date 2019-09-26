
#
#
#
#
#   date: 2019-08-20
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019   Giuseppe Puglisi    gpuglisi@stanford.edu
#



import numpy as np
import tensorflow as tf
import neuralgym as ng
from .generative_inpainting_model import InpaintCAModel


from utils import MinMaxRescale

class ContextualAttention(InpaintCAModel) :
    def __init__(self, modeldir  = None , verbose=False   ):
        self.checkpoint_dir = modeldir
        self.verbose=verbose

        super(ContextualAttention , self).__init__()

    def setup_input(self,fname_masked    ):
        maskdmap=np.load(fname_masked)
        holemask = np.ma.masked_not_equal(maskdmap,0) .mask
        maxval = maskdmap[holemask].max() ; minval = maskdmap[holemask].min()
        maskdmap = MinMaxRescale(maskdmap ,a=-1, b=1 ) #rescaling to -1,1
        #maskdmap = 2. *(maskdmap -minval) / (maxval - minval) -1
        #maskdmap = (maskdmap - maskdmap.mean()) /maskdmap.std()
        #self.mean = maskdmap.mean() ; self.std = maskdmap.std()
        self.X = maskdmap;
        self.mask  = 1. - np.int_(holemask  )
        self.min = minval;  self.max = maxval

        pass

    def rescale_back (self, v ) :
        return MinMaxRescale(v, a= self.min , b = self.max )
        #return v *self.std + self.mean
        #return ((v+1 )* (self.max - self.min)/2. +
        #           self.min  )

    def preprocess_input ( self  ) :

        self.h, self.w = self.X.shape

        grid = 8
        image  = np.repeat(self.X [:,:,np.newaxis], 3, axis=2)
        mask  = np.repeat( self. mask[:,:,np.newaxis], 3, axis=2)
        #eventually resize the image

        image = image[:self.h//grid*grid, :self.w//grid*grid, :]
        mask = mask[:self.h//grid*grid, :self.w//grid*grid, :]

        if self.verbose  : print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image =  np.concatenate([image, mask], axis=2)
        input_image = tf.constant(input_image, dtype=tf.float32)
        return input_image


    def postprocess_output ( self, output,sess  ):


        output = tf.reverse(output, [-1])

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                self.checkpoint_dir,
                from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        if self.verbose  : print('ContextualAttention Model loaded.')
        result = sess.run(output)
        outarray = result[0][:, :, ::-1].mean(axis=-1,
                    keepdims=True ) .reshape(self.h,self.w)
        return outarray

    def predict (self,   reuse  ):

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = self.preprocess_input( )
            output   = self.build_server_graph(input_image, reuse=reuse  )
            out = self.postprocess_output(output, sess )
            return out
