#
#
#
#
#   date: 2019-08-20
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019   Giuseppe Puglisi    gpuglisi@stanford.edu
#



#import .deep_prior_inpainter as dp
#import .contextual_attention_gan as ca
#import .nearest_neighbours_inpainter as nn

from inpainters import (
  deep_prior_inpainter as dp ,
  contextual_attention_gan    as ca,
  nearest_neighbours_inpainter as nn
  )


class HoleInpainter(object) :

    """
    This class provides an interface to the 3 inpainting techniques.
    One of the key parameters is `args`  importing arguments  input by  the user in the inpainting scripts.
    """

    def __init__ (self, args , Npix = 128, meshgrid=True ) :
        """
        Initialize inpainter with the method given in ``args.method``.
        So far the Deep-Prior and GAN architecture are compatible to run on  ``128x128`` images.


        """
        if args.method =='Deep-Prior':

            self.Inpainter = dp.DeepPrior ( (Npix, Npix, 4),
                                            verbose = args.debug, meshgrid=meshgrid   )
            self.epochs =args.dp_epochs
            self.optimizer="Adam"
            self.Inpainter.compile(optimizer=self.optimizer )

        elif args.method=='Contextual-Attention' :
            self.Inpainter = ca.ContextualAttention( modeldir =args.checkpoint_dir
                        , verbose = args.debug  )

        elif args.method=='Nearest-Neighbours' :
            self.Inpainter = nn.NearestNeighbours(verbose = args.debug, Npix=Npix, tol =args.nn_tol  )
        self.method = args.method
        pass

    def __call__(self, reuse ) :
        """
        Run inpainting,

        **Parameters**

        - `reuse`:{bool}
            whether to recompile or not the Deep-Prior and GAN  neural network.
        """
        if self.method== 'Deep-Prior':
            return self.DPinpaint(reuse=reuse )
        elif self.method== 'Contextual-Attention':
            return self.GANinpaint(reuse=reuse )
        elif self.method== 'Nearest-Neighbours':
            return self.NNinpaint()

    def setup_input(self , fname, rdseed=None  ) :
        """
        Pre-process the  flat map  by renormalizing and reshaping  it
        as it required by the inpainting method
        """
        self.Inpainter.rdseed = rdseed
        return   self.Inpainter.setup_input( fname )

    def DPinpaint(self,reuse ) :

        """
        Set of instructions to inpaint with :class:`DeepPrior`
        """
        if reuse :
            self.Inpainter.compile (optimizer=self.optimizer)

        self.Inpainter.train(self.Inpainter.Z , self.Inpainter.X , epochs=self.epochs )
        self.Inpainter.evaluate(self.Inpainter.Z,self.Inpainter.X)
        p =   self.Inpainter.predict()[0,:,:,0]
        p = self.Inpainter.rescale_back(p )
        return p

    def GANinpaint  (self , reuse  ) :
        """
        Set of instructions to inpaint with
        :class:`ContextualAttention`
        """
        p = self.Inpainter.predict( reuse )
        p = self.Inpainter.rescale_back(p )

        return  p

    def NNinpaint  (self  ) :
        """
        Set of instructions to inpaint with :class:`NearestNeighbours`
        """
        return  self.Inpainter.predict ( )
