from  inpainters  import (
  deep_prior_inpainter as dp ,
  contextual_attention_gan    as ca,
  nearest_neighbours_inpainter as nn
  )
from utils import numpy2png


class HoleInpainter() :
    def __init__ (self, args , Npix = 128 ) :
        if args.method =='Deep-Prior':

            self.Inpainter = dp.DeepPrior ( (Npix, Npix, 1),
                                            verbose = args.debug  )
            self.epochs =args.dp_epochs
            Adaopt="Adam"
            self.Inpainter.compile(optimizer=Adaopt )

        elif args.method=='Contextual-Attention' :
            self.Inpainter = ca.ContextualAttention( modeldir =args.checkpoint_dir
                        , verbose = args.debug  )

        elif args.method=='Nearest-Neighbours' :
            self.Inpainter = nn.NearestNeighbours(verbose = args.debug, Npix=Npix, tol =args.nn_tol  )
        self.method = args.method
        pass

    def __call__(self, reuse ) :
        if self.method== 'Deep-Prior':
            return self.DPinpaint()
        elif self.method== 'Contextual-Attention':
            return self.GANinpaint(reuse=reuse )
        elif self.method== 'Nearest-Neighbours':
            return self.NNinpaint()

    def setup_input(self , fname ) :
        return   self.Inpainter.setup_input( fname )



    def rescale_back (self, v ) :
        return  ( v* (self.Inpainter.max - self.Inpainter.min) +
                    self.Inpainter.min )


    def DPinpaint(self) :

        self.Inpainter.train(self.Inpainter.Z , self.Inpainter.X , epochs=self.epochs )
        self.Inpainter.evaluate(self.Inpainter.Z,self.Inpainter.X)
        # predict and rescale back
        p =   self.Inpainter.predict()[0,:,:,0]
        p = self.rescale_back(p )
        return p

    def GANinpaint  (self , reuse  ) :
        image = numpy2png(self.Inpainter.X )
        mask = numpy2png (1 - self.Inpainter.mask )

        p = self.Inpainter.predict(image, mask , reuse )
        p = self.rescale_back(p )

        return  p

    def NNinpaint  (self  ) :
         return  self.Inpainter.predict ( )
