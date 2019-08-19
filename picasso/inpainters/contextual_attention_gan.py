import numpy as np
import tensorflow as tf
import neuralgym as ng
from .generative_inpainting_model import InpaintCAModel

class ContextualAttention(InpaintCAModel) :
    def __init__(self, modeldir  = None , verbose=False   ):
        self.checkpoint_dir = modeldir
        self.verbose=verbose

        super(ContextualAttention , self).__init__()

    def setup_input(self,fname_masked    ):
        maskdmap=np.load(fname_masked)
        holemask = np.ma.masked_not_equal(maskdmap,0) .mask
        maxval = maskdmap[holemask].max() ; minval = maskdmap[holemask].min()
        maskdmap = (maskdmap -minval) / (maxval - minval)
        self.X = maskdmap;
        self.mask  = np.int_(holemask  )
        self.min = minval;  self.max = maxval
        pass

    def preprocess_input ( self, image, mask ) :
        self.h, self.w,_  = image.shape
        grid = 8
        image = image[:self.h//grid*grid, :self.w//grid*grid, :]
        mask = mask[:self.h//grid*grid, :self.w//grid*grid, :]

        if self.verbose  : print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image =  np.concatenate([image, mask], axis=2)
        input_image = tf.constant(input_image, dtype=tf.float32)
        return input_image


    def postprocess_output ( self, output,sess  ):

        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
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
                    keepdims=1) .reshape(self.h,self.w)
        return outarray /255.


    def predict (self, image , mask ):

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = self.preprocess_input( image, mask )
            output   = self.build_server_graph(input_image)
            out = self.postprocess_output(output, sess )
            return out
