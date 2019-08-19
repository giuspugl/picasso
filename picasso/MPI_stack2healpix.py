import healpy as hp
import numpy as np
import argparse
from mpi4py import MPI

from  inpainters  import (
  deep_prior_inpainter as dp ,
  contextual_attention_gan    as ca,
  )

from utils import utils

from  utils import (
    setup_input,
    set_header,
    f2h,
    rd2tp,
    numpy2png

)






class HoleInpainter() :
    def __init__ (self, method , Npix = 128, modeldir = None, verbose= False  ) :
        if method =='Deep-Prior':
            self.Inpainter = dp.DeepPrior ( (Npix, Npix, 1), verbose = verbose )
            self.epochs = 2#000
            Adaopt="Adam"
            self.Inpainter.compile(optimizer=Adaopt )
            self.exec  = self.DPinpaint
        elif method=='Contextual-Attention' :
            self.Inpainter = ca.ContextualAttention( modeldir =modeldir , verbose = verbose )
            self.exec  = self.GANinpaint

        pass


    def setup_input(self , fname ) :
        return   self.Inpainter.setup_input( fname )



    def rescale_back (self, v ) :
        return  ( v* (self.Inpainter.max - self.Inpainter.min) +
                    self.Inpainter.min )


    def DPinpaint(self     ) :

        self.Inpainter.train(self.Inpainter.Z , self.Inpainter.X , epochs=self.epochs )
        self.Inpainter.evaluate(self.Inpainter.Z,self.Inpainter.X)
        # predict and rescale back
        p =   self.Inpainter.predict(self.Inpainter.Z)
        p = self.rescale_back(p )
        return p

    def GANinpaint  (self  ) :
        image = numpy2png(self.Inpainter.X )
        mask = numpy2png (1 - self.Inpainter.mask )

        p = self.Inpainter.predict(image, mask )
        p = self.rescale_back(p )

        return  p


def main(args):
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    Npix = 128 ## WARNING: This is hard-coded because of the architecture of both CNN

    glob_ra,glob_dec, _  = np.loadtxt(args.ptsourcefile ,unpack=True)

    localsize =np.int_(  glob_ra.shape[0]/nprocs  ) ## WARNING:  this MUST  evenly divide!!!!!!

    ra =  glob_ra[slice( rank *localsize ,  (rank +1)* localsize)]
    dec =  glob_dec[slice( rank *localsize ,  (rank +1)* localsize)]
    Nstacks= ra.shape [0]

    if args.pol :
        keys = ['T', 'Q', 'U']
        inputmap = hp.read_map(args.hpxmap  ,field=[0,1,2] )
    else:
        keys = ['T' ]
        inputmap = [hp.read_map( args.hpxmap) ]


    mask = np.zeros_like (inputmap[0] )

    nside = hp.get_nside(inputmap)

    size_im = {2048: 192.  ,4096 : 64., 32 :360. }
    beam =np.deg2rad( args.beamsize /60.)

    Inpainter =  HoleInpainter (args.method,
                    modeldir = args.checkpoint_dir,
                    verbose  =args.debug )

    for i in range(Nstacks):

        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )
        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,3* beam)
        mask [pixs]  = 1.
        for k,j  in  zip(keys, range(len(inputmap)) ) :
            fname = args.stackfile+k+'_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] )
            fname = args.stackfile

            Inpainter.setup_input( fname  )
            predicted = Inpainter.exec ()

            np.save(args.stackfile+k+'_{:.5f}_{:.5f}_inpainted.npy'.format(ra[i],dec[i] ), predicted)
            maskmap =  f2h (predicted ,header, nside )
            inputmap[j][pixs] = inpaintedmap[pixs]
        break

        maps  = np.concatenate(inputmap).reshape(hp.nside2npix(nside), len(inputmap))
        reducmaps = np.zeros_like(maps)
        globmask= np.zeros_like(mask)

        comm.Allreduce(maps, reducmaps, op=MPI.SUM)
        comm.Allreduce(mask, globmask , op=MPI.SUM)
        if rank ==0 :
            hp.write_map(args.inpaintedmap , [inputmap[k] *(1- globmask) + reducmaps[:,k]  *globmask for k in range(len(inputmap))]  )

        comm.Barrier()

        comm.Disconnect



if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked, no extension ' )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the file with stacked maps')
	parser.add_argument("--ptsourcefile", help='path to the file with RA, Dec coordinates of sources to be inpainted ')
	parser.add_argument("--inpaintedmap", help='path to the inpainted HEALPIX map  ')
	parser.add_argument("--method", help=" string of inpainting technique, can be 'Deep-Prior', 'Contextual-Attention'. ")
	parser.add_argument("--pol", action="store_true" , default=False )
	parser.add_argument('--checkpoint_dir', default='', type=str,help='The directory of tensorflow checkpoint for the ContextualAttention.')
	parser.add_argument('--debug', default=False , action='store_true')
	args = parser.parse_args()
	main( args)
