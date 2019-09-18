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
from mpi4py import MPI


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
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    Npix = 128 ## WARNING: This is hard-coded because of the architecture of both CNN

    glob_ra,glob_dec   = np.loadtxt(args.ptsourcefile ,unpack=True)
    localsize = np.int_(glob_ra.shape[0]/nprocs)
    remainder = glob_ra.shape[0]% nprocs
    if (rank < remainder) :
    #  The first 'remainder' ranks get 'count + 1' tasks each
        start = np.int_(rank * (localsize  + 1))
        stop =np.int_(  start + localsize +1  )
    else:
    # The remaining 'size - remainder' ranks get 'count' task each
        start = np.int_(rank * localsize + remainder  )
        stop =np.int_(  start + (localsize  )   )

    ra =  glob_ra[slice( start , stop )]
    dec =  glob_dec[slice( start , stop )]
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


    for i in range(Nstacks):
        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )
        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,3* beam)
        mask [pixs]  = 1.
        for k,j  in  zip(keys, range(len(inputmap)) ) :
            fname = args.stackfile+k+'_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] )

            Inpainter.setup_input( fname  )

            predicted = Inpainter ()
            np.save(args.stackfile+k+'_{:.5f}_{:.5f}{}.npy'.format(ra[i],dec[i],args.method ), predicted)
            inpaintedmap, footprint =  f2h (predicted ,header, nside )
            inputmap[j][pixs] = inpaintedmap[pixs]

        if i ==1:  break

    maps  = np.concatenate(inputmap*mask ).reshape(hp.nside2npix(nside), len(inputmap))
    reducmaps = np.zeros_like(maps)
    globmask= np.zeros_like(mask)
    comm.Allreduce(maps , reducmaps, op=MPI.SUM)
    comm.Allreduce(mask, globmask , op=MPI.SUM)
    if rank ==0 and args.outputmap :
        hp.write_map(args.outputmap , [inputmap[k] *(1- globmask) + reducmaps[:,k]  *globmask for k in range(len(inputmap))] , overwrite=args.overwrite    )

    comm.Barrier()

    comm.Disconnect



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
	parser.add_argument('--nearest-neighbours-iters' , dest = 'nn_iters', type= np.int, default = 100 )
	parser.add_argument('--overwrite', default=False , action='store_true')

	parser.add_argument('--debug', default=False , action='store_true')

	args = parser.parse_args()
	main( args)


"""
mpirun -np 2  python picasso/picasso/MPI_stack2healpix.py  --stackfile ~/work/inpainting/T_99.89973_-51.09838_masked.npy \
 --ptsourcefile ~/work/inpainting/FG_inpainting/pccs_857_planck.dat --outputmap test.fits \
 --overwrite --debug --hpxmap  ~/work/heavy_maps/HFI_SkyMap_857-field-Int_2048_R3.00_full.fits  \
 --beamsize 5 --deep-prior-epochs 10 --checkpoint_dir  /Users/peppe/work/inpainting/FG_inpainting/model_logs/cmb  \
 --method Deep-Prior
"""
