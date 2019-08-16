import healpy as hp
import numpy as np
import argparse
from mpi4py import MPI
from utils import (
    setup_input,
    f2h,
    rd2tp

)



from deep_prior_inpainter import DeepPrior

class HoleInpainter() :
    def __init__ (self, method , Npix = 128) :
        if method =='DeepPrior':
            self.Inpainter = DeepPrior ( (Npix, Npix, 1))
            self.epochs = 2#000
            Adaopt="Adam"
            self.Inpainter.compile(optimizer=Adaopt )
        elif method=='WGAN' :
            self.Inpainter =None
        pass

    def __call__(self, X ,Z, min,max ) :
        self.Inpainter.train(Z , X , epochs=self.epochs )
        self.Inpainter.evaluate(Z,X)
        # predict and rescale back
        p = self.Inpainter.predict(Z) * (max - min) + min
        return p

def main(args):

    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    Npix = 128
    """
    glob_ra,glob_dec, _  = np.loadtxt(args.ptsourcefile ,unpack=True)

    localsize = glob_ra.shape[0]/nprocs  ## WARNING:  this MUST  evenly divide!!!!!!

    ra =  glob_ra[slice( rank *localsize ,  (rank +1)* localsize)]
    dec =  glob_dec[slice( rank *localsize ,  (rank +1)* localsize)]
    Nstacks= ra.shape [0]

    if args.pol :
        keys = ['T', 'Q', 'U']
        sima = hp.read_map(args.hpxmap  ,field=[0,1,2] )
    else:
        keys = ['T' ]
        sima = [hp.read_map( args.hpxmap) ]


    mask = np.zeros_like (sima[0] )

    nside = hp.get_nside(sima)
    """
    size_im = {2048: 192.  ,4096 : 64. }
    beam =np.deg2rad( args.beamsize /60.)

    Inpainter =  HoleInpainter (args.method)
    for i in range(1):
        fname = args.stackfile
        maskdmap, noisemap ,minval, maxval = setup_input( fname)

        predicted = Inpainter (maskdmap, noisemap, minval,maxval )
    """
    for i in range(Nstacks):

        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )
        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,3* beam)
        mask [pixs]  = 1.
        for k,j  in  zip(keys, range(len(sima)) ) :
    		fname = args.stackfile+k+'_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] )

    		maskdmap, noisemap ,minval, maxval = setup_input( fname)
    		predicted = Inpainter (maskdmap, noisemap, minval,maxval )

 	        np.save(args.stackfile+k+'_{:.5f}_{:.5f}_inpainted.npy'.format(ra[i],dec[i] ), predicted)
  	    	maskmap =  f2h (predicted ,header, nside )
	        sima[j][pixs] = inpaintedmap[pixs]
		break

        """

    maps  = np.concatenate(sima).reshape(hp.nside2npix(nside), len(sima))
    reducmaps = np.zeros_like(maps)
    globmask= np.zeros_like(mask)

    comm.Allreduce(maps, reducmaps, op=MPI.SUM)
    comm.Allreduce(mask, globmask , op=MPI.SUM)
    if rank ==0 :
        hp.write_map(args.inpaintedmap , [sima[k] *(1- globmask) + reducmaps[:,k]  *globmask for k in range(len(sima))]  )

    comm.Barrier()

    comm.Disconnect



if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked, no extension ' )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the file with stacked maps')
	parser.add_argument("--ptsourcefile", help='path to the file with RA, Dec coordinates of sources to be inpainted ')
	parser.add_argument("--inpaintedmap", help='path to the inpainted HEALPIX map  ')
	parser.add_argument("--method", help=" string of inpainting technique, can be 'DeepPrior', 'WGAN'. ")
	parser.add_argument("--pol", action="store_true" , default=False )
	args = parser.parse_args()
	main( args)
