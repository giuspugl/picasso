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
import os
import sys
#print (sys.version)

codename = (r"""

8888888b. 8888888 .d8888b.        d8888  .d8888b.   .d8888b.   .d88888b.
888   Y88b  888  d88P  Y88b      d88888 d88P  Y88b d88P  Y88b d88P" "Y88b
888    888  888  888    888     d88P888 Y88b.      Y88b.      888     888
888   d88P  888  888           d88P 888  "Y888b.    "Y888b.   888     888
8888888P"   888  888          d88P  888     "Y88b.     "Y88b. 888     888
888         888  888    888  d88P   888       "888       "888 888     888
888         888  Y88b  d88P d8888888888 Y88b  d88P Y88b  d88P Y88b. .d88P
888       8888888 "Y8888P" d88P     888  "Y8888P"   "Y8888P"   "Y88888P"

""" )

from  inpainters  import (
  deep_prior_inpainter as dp ,
  contextual_attention_gan    as ca,
  nearest_neighbours_inpainter as nn,
  interfaces
  )
from inpainters.interfaces  import HoleInpainter

from utils import utils

from  utils import (
    set_header,
    f2h,
    rd2tp,

)



def main(args):
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    ( np.linspace (-1,1,  rank*10000 ))**0.5
    if rank==0 :
        print(codename)
    nprocs  = comm.Get_size()
    Npix = 128 ## WARNING: This is hard-coded because of the architecture of both CNN
    try :
        os.makedirs(args.outdir+f"{args.method}")
    except   FileExistsError:
        print (f"Warning: Overwriting files in {args.outdir}{args.method}")

    try :
        glob_ra,glob_dec  = np.loadtxt(args.ptsourcefile ,unpack=True)[-10:, -10:]
    except ValueError:
        glob_ra,glob_dec  = np.loadtxt(args.ptsourcefile ,unpack=False)


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

    if args.pol and args.skipT  :
        if rank ==0  :print ("Skipping T, inpainting Q, U ")
        keys = ['Q', 'U']
        inputmap = hp.read_map(args.hpxmap  ,field=[1,2] ,verbose=args.debug)
    elif args.pol and not args.skipT :
        keys = ['T', 'Q', 'U']
        if rank ==0  :  print ("Inpainting T,Q,U")
        inputmap = hp.read_map(args.hpxmap  ,field=[0,1,2] ,verbose=args.debug)

    elif not args.skipT and not args.pol :
        if rank ==0  :  print ("Inpainting T ")
        keys = ['T' ]
        inputmap = [hp.read_map( args.hpxmap, verbose=args.debug) ]
    else:
        raise ValueError (f'Please indicate  what you wanna inpaint'
                            'with input arguments    --skip-temperature and  --pol'
                            'at least one of them needs to be True, '
                            'they are  {args.skipT} {args.pol}' )
        return




    mask = np.zeros_like (inputmap[0] )

    nside = hp.get_nside(inputmap)

    size_im = {2048: 192.  ,4096 : 64., 1024:384. }

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

            Inpainter.setup_input( fname , rdseed =(i +129292) )

            predicted = Inpainter (reuse =reuse )
            np.save(args.outdir+args.method +'/'+k+'_{:.5f}_{:.5f}.npy'.format( ra[i],dec[i]), predicted)
            inpaintedmap, footprint =  f2h (predicted ,header, nside )
            inputmap[j][pixs] = inpaintedmap[pixs]
            if not reuse : reuse =True

 

    maps  = np.concatenate(inputmap*mask ).reshape(hp.nside2npix(nside), len(inputmap))
    reducmaps = np.zeros_like(maps)
    globmask= np.zeros_like(mask)
    comm.Allreduce(maps , reducmaps, op=MPI.SUM)
    comm.Allreduce(mask, globmask , op=MPI.SUM)
    if rank ==0 and args.outputmap :
        hp.write_map(args.outputmap , [inputmap[k] *(1- globmask) + reducmaps[:,k]  *globmask for k in range(len(inputmap))]
                    , overwrite=args.overwrite    )

    comm.Barrier()

    comm.Disconnect



if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked, no extension ' )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the directory with stacked masked maps')
	parser.add_argument("--outdir", help='path to the outputs with stacked inpainted  maps')
	parser.add_argument("--ptsourcefile", help='path to the file with RA, Dec coordinates of sources to be inpainted ')
	parser.add_argument("--outputmap", help='path and name  to the inpainted HEALPIX map  ')
	parser.add_argument("--method", help=" string of inpainting technique, can be 'Deep-Prior', 'Contextual-Attention', 'Nearest-Neighbours'. ")
	parser.add_argument("--pol", action="store_true" , default=False )
	parser.add_argument("--skip_temperature",dest ="skipT", action="store_true" , default=False )

	parser.add_argument('--checkpoint_dir', default='', type=str,help='The directory of tensorflow checkpoint for the ContextualAttention.')
	parser.add_argument('--deep-prior-epochs',dest='dp_epochs',  type= np.int, default = 2000)
	parser.add_argument('--nearest-neighbours-tolerance' , dest = 'nn_tol', type= np.float, default = 1e-8 )
	parser.add_argument('--overwrite', default=False , action='store_true')

	parser.add_argument('--debug', default=False , action='store_true')

	args = parser.parse_args()
	main( args)


"""
mpirun -np 4  python ~/work/picasso/picasso/inpaint_mpi.py  --stackfile ~/work/inpainting/stacks/synch/singlestacks/ \
--ptsourcefile ~/work/inpainting/FG_inpainting/ptsrcS3_2019-08-02.dat  --outdir outputs/synch/  \
--outputmap ~/work/heavy_maps/test.fits --hpxmap ~/work/heavy_maps/SPASS_pysm_s1d1_10arcmin.fits  \
--beamsize 10 --deep-prior-epochs 10 \
 --checkpoint_dir  /Users/peppe/work/inpainting/model_logs/synch --method Deep-Prior \
 --overwrite --debug
"""
