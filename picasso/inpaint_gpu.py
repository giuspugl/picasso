#
#
#
#
#   date: 2019-08-20
#   author: GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019   Giuseppe Puglisi    gpuglisi@stanford.edu
#

import healpy as hp
import numpy as np
import argparse


from  inpainters  import (
  deep_prior_inpainter as dp ,
  contextual_attention_gan    as ca,
  nearest_neighbours_inpainter as nn,
interfaces
)
from inpainters.interfaces  import HoleInpainter


from utils import utils

from  utils import (
    setup_input,
    set_header,
    f2h,
    rd2tp,
    numpy2png

)



def main(args):

    Npix = 128 ## WARNING: This is hard-coded because of the architecture of both CNN

    ra,dec   = np.loadtxt(args.ptsourcefile ,unpack=True)

    Nstacks= ra.shape [0]

    if args.pol :
        keys = ['T', 'Q', 'U']
        inputmap = hp.read_map(args.hpxmap  ,field=[0,1,2] ,verbose=args.debug)
    else:
        keys = ['T' ]
        inputmap = [hp.read_map( args.hpxmap, verbose=args.debug) ]


    mask = np.zeros_like (inputmap[0] )

    nside = hp.get_nside(inputmap)

    size_im = {2048: 192.  ,4096 : 64., 32 :360. }
    beam =np.deg2rad( args.beamsize /60.)

    Inpainter =  HoleInpainter (args , Npix=Npix  )

    reuse = False
    for i in range(Nstacks):
        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )
        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,3* beam)
        mask [pixs]  = 1.
        for k,j  in  zip(keys, range(len(inputmap)) ) :

            fname = args.stackfile+k+'_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] )
            Inpainter.setup_input( fname  , rdseed =(i +129292) )
            import time
            s= time.clock()
            predicted = Inpainter(reuse=reuse  )
            e = time.clock()
            print(e-s)
            np.save(args.stackfile+k+'_{:.5f}_{:.5f}{}.npy'.format(ra[i],dec[i],args.method ), predicted)
            inpaintedmap, footprint =  f2h (predicted ,header, nside )

            inputmap[j][pixs] = inpaintedmap[pixs]

            if not reuse : reuse =True

    if args.outputmap :
        hp.write_map(args.outputmap , [inputmap[k]  for k in range(len(inputmap))] , overwrite=args.overwrite    )


if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked, no extension ' )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the file with stacked maps')
	parser.add_argument("--ptsourcefile", help='path to the file with RA, Dec coordinates of sources to be inpainted ')
	parser.add_argument("--outputmap", help='path and name  to the inpainted HEALPIX map  ')
	parser.add_argument("--method", help=" string of inpainting technique, can be 'Deep-Prior', 'Contextual-Attention', 'Nearest-Neighbours'. ")
	parser.add_argument("--pol", action="store_true" , default=False )
	parser.add_argument('--checkpoint_dir', default='', type=str,help='The directory of tensorflow checkpoint for the ContextualAttention.')
	parser.add_argument('--deep-prior-epochs',dest='dp_epochs',  type= np.int, default = 2000)
	parser.add_argument('--nearest-neighbours-tolerance' , dest = 'nn_tol', type= np.float, default = 1e-8 )

	parser.add_argument('--overwrite', default=False , action='store_true')

	parser.add_argument('--debug', default=False , action='store_true')

	args = parser.parse_args()
	main( args)
