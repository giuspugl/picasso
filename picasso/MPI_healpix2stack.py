import healpy as hp
import numpy as np
import argparse
from mpi4py import MPI

from  utils import utils

from  utils import (
    setup_input,
    h2f,
    set_header,
    rd2tp

)



def main(args):

    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()
    glob_ra,glob_dec, _  = np.loadtxt(args.ptsourcefile ,unpack=True)

    localsize = glob_ra.shape[0]/nprocs  ## WARNING:  this MUST  evenly divide!!!!!!

    ra =  glob_ra[slice( rank *localsize ,  (rank +1)* localsize)]
    dec =  glob_dec[slice( rank *localsize ,  (rank +1)* localsize)]

    Nstacks= ra.shape [0]
    Npix = 128 #This is hard-coded because of the architecture of both CNN
    if args.pol :
        keys = ['T', 'Q', 'U']
        inputmap = hp.read_map(args.hpxmap  ,field=[0,1,2] )
    else:
        keys = ['T' ]

        inputmap = [hp.read_map( args.hpxmap) ]

    mask = np.ones_like (inputmap[0] )
    beam =np.deg2rad( args.beamsize /60.)


    nside = hp.get_nside(inputmap)
    size_im = {2048: 192.  ,4096 : 64. }
    for i in range(Nstacks):
        sizepatch = size_im[nside]*1. /Npix/60.
        header       = set_header(ra[i],dec[i], sizepatch )

        tht,phi      = rd2tp(ra[i],dec[i])
        vec          = hp.ang2vec( theta = tht,phi =phi )
        pixs         = hp.query_disc(nside,vec,3* beam)
        mask[pixs]   = 0

        for k,j  in  zip(keys, range(len(inputmap)) ) :
        	np.save(args.stackfile+k+'_{:.5f}_{:.5f}_masked'.format(ra[i],dec[i] ),
                                h2f(mask * inputmap[j] ,header))
        	np.save(args.stackfile+k+'_{:.5f}_{:.5f}'.format(ra[i],dec[i])  , h2f(inputmap[j] ,header) )

        if i %100 ==0  and rank ==0  :
	    	print("Stacking %d source "%i   )


    comm.Barrier()

    if rank ==0 :
        print ("collecting stacks to 1 single file" )
        globT = np.zeros( (Nstacks, 128,128))
        mglobT = np.zeros( (Nstacks, 128,128))
        if args.pol :
        	globQ = np.zeros( (Nstacks, 128,128))
        	globU = np.zeros( (Nstacks, 128,128))
        	mglobQ = np.zeros( (Nstacks, 128,128))
	        mglobU = np.zeros( (Nstacks, 128,128))

        for i in range(Nstacks):
            globT[i,:,: ] =np.load (args.stackfile+ 'T_{:.5f}_{:.5f}.npy'.format(ra[i],dec[i] ))
            mglobT[i,:,: ] =np.load (args.stackfile+ 'T_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] ))
            if args.pol:
                globQ[i,:,: ] =np.load (args.stackfile+ 'Q_{:.5f}_{:.5f}.npy'.format(ra[i],dec[i] ))
            	mglobQ[i,:,: ] =np.load (args.stackfile+ 'Q_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] ))
            	globU[i,:,: ] =np.load (args.stackfile+ 'U_{:.5f}_{:.5f}.npy'.format(ra[i],dec[i] ))
                mglobU[i,:,: ] =np.load (args.stackfile+ 'U_{:.5f}_{:.5f}_masked.npy'.format(ra[i],dec[i] ))
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
	parser.add_argument("--stackfile", help='path to the file with stacked maps')
	parser.add_argument("--pol", action="store_true" , default=False )
	args = parser.parse_args()
	main( args)
