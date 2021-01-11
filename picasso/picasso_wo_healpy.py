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
import os
import glob
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


#from utils import utils

#from  utils import (
    #set_header,
    #f2h,
#    rd2tp,

#)
import warnings
warnings.filterwarnings("ignore")


def main(args):

    Npix = 128 ## WARNING: This is hard-coded because of the architecture of both CNN
    if args.debug :
        print(codename )
    try :
        os.makedirs(f"{args.outdir}/{args.method}")
    except  FileExistsError:
        print (f"Warning: Overwriting files in {args.outdir}/{args.method}")


    beam =np.deg2rad( args.beamsize /60.)
    Inpainter =  HoleInpainter (args , Npix=Npix  )
    reuse = False
    files=  glob.glob(f"{args.stackfile}/*.npy")[:args.Ninpaints]


    for i, fname  in enumerate ( files):
        # loops  over the *.npy files located in the folder given in
        # the input argument args.stackfile
        # outputs the inpainted results in the following directory

        outfile =f"{args.outdir}/{args.method}/{fname.split('/')[-1]}"
        if os.path.exists(outfile ) and not args.overwrite  :
            print("File exists, skipping")

        Inpainter.setup_input( fname  , rdseed =(i +129292) )
        predicted = Inpainter(reuse=reuse  )
        np.save(outfile , predicted)


        if not reuse :
            reuse =True
        if args.method =='Deep-Prior':
            Inpainter =  HoleInpainter (args , Npix=Npix  )


if __name__=="__main__":
	parser = argparse.ArgumentParser( description=" inpainting with GPUs from a healpix map. Usage example: "
    )
	parser.add_argument("--beamsize", help = 'beam size in arcminutes of the input map', type=np.float  )
	parser.add_argument("--stackfile", help='path to the directory with stacked masked maps')
	parser.add_argument("--outdir", help='path to the outputs with stacked inpainted  maps')
	parser.add_argument("--method", help=" string of inpainting technique, can be 'Deep-Prior', 'Contextual-Attention', 'Nearest-Neighbours'. ")
	parser.add_argument('--checkpoint_dir', default='', type=str,help='The directory of tensorflow checkpoint for the ContextualAttention.')
	parser.add_argument('--deep-prior-epochs',dest='dp_epochs',  type= np.int, default = 2000)
	parser.add_argument('--nearest-neighbours-tolerance' , dest = 'nn_tol', type= np.float, default = 1e-8 )
	parser.add_argument('--Ninpaints' ,   type= np.int, default = 0, help='useful for debugging, you can perform a smaller number of inpaintings  ' )
	parser.add_argument('--overwrite', default=False , action='store_true')
	parser.add_argument('--debug', default=False , action='store_true', help='set the level of ouput verbosity ')

	args = parser.parse_args()
	main( args)
