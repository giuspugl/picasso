from  inpainters  import (
  deep_prior_inpainter as dp ,
  contextual_attention_gan    as ca,
  nearest_neighbours_inpainter as nn
  )


class HoleInpainter() :
    def __init__ (self, args , Npix = 128 ) :
        if args.method =='Deep-Prior':

            self.Inpainter = dp.DeepPrior ( (Npix, Npix, 1),
                                            verbose = args.debug  )
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
        if self.method== 'Deep-Prior':
            return self.DPinpaint(reuse=reuse )
        elif self.method== 'Contextual-Attention':
            return self.GANinpaint(reuse=reuse )
        elif self.method== 'Nearest-Neighbours':
            return self.NNinpaint()

    def setup_input(self , fname, rdseed=None  ) :
        self.Inpainter.rdseed = rdseed
        return   self.Inpainter.setup_input( fname )

    def DPinpaint(self,reuse ) :
        if reuse :
            self.Inpainter.compile (optimizer=self.optimizer)
        self.Inpainter.train(self.Inpainter.Z , self.Inpainter.X , epochs=self.epochs )
        self.Inpainter.evaluate(self.Inpainter.Z,self.Inpainter.X)
        # predict and rescale back
        p =   self.Inpainter.predict()[0,:,:,0]
        p = self.Inpainter.rescale_back(p )
        return p

    def GANinpaint  (self , reuse  ) :

        p = self.Inpainter.predict( reuse )
        p = self.Inpainter.rescale_back(p )

        return  p

    def NNinpaint  (self  ) :
         return  self.Inpainter.predict ( )
