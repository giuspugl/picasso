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
import os
import argparse
from mpi4py import MPI

from  utils import utils

from  utils import (
    h2f,
    set_header,
    rd2tp
)

print(r"""

8888888b. 8888888 .d8888b.        d8888  .d8888b.   .d8888b.   .d88888b.
888   Y88b  888  d88P  Y88b      d88888 d88P  Y88b d88P  Y88b d88P" "Y88b
888    888  888  888    888     d88P888 Y88b.      Y88b.      888     888
888   d88P  888  888           d88P 888  "Y888b.    "Y888b.   888     888
8888888P"   888  888          d88P  888     "Y88b.     "Y88b. 888     888
888         888  888    888  d88P   888       "888       "888 888     888
888         888  Y88b  d88P d8888888888 Y88b  d88P Y88b  d88P Y88b. .d88P
888       8888888 "Y8888P" d88P     888  "Y8888P"   "Y8888P"   "Y88888P"

""" )

def main(args):

    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()

    try :
        os.makedirs(args.stackfile+ 'singlestacks')
    except  FileExistsError:
        print (f"Warning: Overwriting files in {args.stackfile+'singlestacks'}")


    try :
        glob_ra,glob_dec  = np.loadtxt(args.ptsourcefile ,unpack=True)
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
    Npix = 128 #This is hard-coded because of the architecture of both CNN
    if args.pol :
        keys = ['T', 'Q', 'U']
        inputmap = hp.read_map(args.hpxmap  ,field=[0,1,2], verbose=False )
    else:
        keys = ['T' ]

        inputmap = [hp.read_map( args.hpxmap, verbose=False ) ]

    mask = np.ones_like (inputmap[0] )
    beam =np.deg2rad( args.beamsize /60.)


    nside = hp.get_nside(inputmap)
    size_im = {2048: 192.  ,4096 : 64., 1024:384. }
    for i in range(Nstacks):
        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )

        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,18* beam)
        mask[pixs]   = 0

        for k,j  in  zip(keys, range(len(inputmap)) ) :
        	np.save(args.stackfile+'singlestacks/'+k+'_{:.5f}_{:.5f}_masked'.format(ra[i],dec[i] ),
                                h2f(mask * inputmap[j] ,header))
        	np.save(args.stackfile+'singlestacks/'+k+'_{:.5f}_{:.5f}'.format(ra[i],dec[i])  , h2f(inputmap[j] ,header) )
        if i %100 ==0  and rank ==0  :
            print("Stacking %d source "%i   )


    comm.Barrier()

    if rank ==0 :
        print ("collecting stacks to 1 single file" )
        glob_Nstacks= glob_ra.shape [0]
        globT = np.zeros( (glob_Nstacks, Npix,Npix))
        mglobT = np.zeros( (glob_Nstacks, Npix,Npix))
        if args.pol :
            globQ = np.zeros( (glob_Nstacks, Npix,Npix))
            globU = np.zeros( (glob_Nstacks, Npix,Npix))
            mglobQ = np.zeros( (glob_Nstacks, Npix,Npix))
            mglobU = np.zeros( (glob_Nstacks, Npix,Npix))
        for i in range(glob_Nstacks):
            #if i%1000 == 0 : print(i)
            globT[i,:,: ] =np.load (args.stackfile+ 'singlestacks/T_{:.5f}_{:.5f}.npy'.format(glob_ra[i],glob_dec[i] ))
            mglobT[i,:,: ] =np.load (args.stackfile+ 'singlestacks/T_{:.5f}_{:.5f}_masked.npy'.format(glob_ra[i],glob_dec[i] ))
            if args.pol:
                globQ[i,:,: ] =np.load (args.stackfile+ 'singlestacks/Q_{:.5f}_{:.5f}.npy'.format(glob_ra[i],glob_dec[i] ))
                mglobQ[i,:,: ] =np.load (args.stackfile+ 'singlestacks/Q_{:.5f}_{:.5f}_masked.npy'.format(glob_ra[i],glob_dec[i] ))
                globU[i,:,: ] =np.load (args.stackfile+ 'singlestacks/U_{:.5f}_{:.5f}.npy'.format(glob_ra[i],glob_dec[i] ))
                mglobU[i,:,: ] =np.load (args.stackfile+ 'singlestacks/U_{:.5f}_{:.5f}_masked.npy'.format(glob_ra[i],glob_dec[i] ))
        np.save(args.stackfile+'T_masked',  mglobT)
        np.save(args.stackfile+'T' , globT)
        if args.pol:
            np.save(args.stackfile+'Q_masked',  mglobQ)
            np.save(args.stackfile +'Q', globQ)
            np.save(args.stackfile+'U_masked',  mglobU)
            np.save(args.stackfile+'U' , globU)
    comm.Barrier()

    comm.Disconnect



if __name__=="__main__":
	parser = argparse.ArgumentParser( description="prepare training and testing dataset from a healpix map " )
	parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked, no extension ' )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the directory where to save stacked maps')
	parser.add_argument("--ptsourcefile", help='path to the file with RA, Dec coordinates of sources to be inpainted ')
	parser.add_argument("--pol", action="store_true" , default=False )
	args = parser.parse_args()
	main( args)
