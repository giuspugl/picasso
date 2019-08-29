#
#
#
#
#   date: 2019-08-20
#   author: XIRAN BAI, GIUSEPPE PUGLISI
#   python3.6
#   Copyright (C) 2019
#
import numpy as np


class NearestNeighbours():
    def __init__ (self, Npix =128 ,   verbose = False ,  tol=1e-8  ) :
        self.verbose=  verbose
        self.niter  = 500
        self.Npix =Npix
        self.tol = tol
        pass


    def setup_input (self, fname_masked  )   :
        self.X =np.load(fname_masked)
        self.mask = np.int_ ( np.ma.masked_not_equal(self.X,0) .mask )
        pass

    def predict (self   ):
        """
        Nearest-neighbours inpaint can now inpaint
        pixels at the border of the image.

        Inpainting loop is stopped when the inpainted array
        in two subsequent iterations is essentially the same with
        a given tolerance threshold `tol`.
        Default tolerance is the one set from numpy.allclose, `1e-8`.
        """

        mask_pos = np.where(self.mask ==0)
        x,y =mask_pos
        p = self.X.copy()
        p[np.logical_not (self.mask )] = self.X [self.mask ] . mean ()
        X=Y =self.Npix
        neighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                                       for y2 in range(y-1, y+2)
                                       if (-1 < x < X and
                                           -1 < y < Y and
                                           (x != x2 or y != y2) and
                                           (0 <= x2 < X) and
                                           (0 <= y2 < Y))]
        while True:
            tmp = p[mask_pos]
            for i,j in zip(  x,y):
                p[i,j] = np.array([p [k,l]  for k,l in  neighbors(i,j ) ] ).mean()


            if    np.allclose(p[mask_pos ] ,  tmp, atol=self.tol ):  break
        return p

    def predict_old (self ):
        '''
        mask: binary mask with the hole as 0, and the rest of as 1
        self.X: input nonmasked data (1 channel)
        niter: number of iterations. The saturation iterations depends on the mask size.
               (a mask size of ~300 takes about ~50 iters)
        '''

        p = self.X.copy()
        mask_pos = np.where(self.mask==0)

        x, y = mask_pos

        p[ (mask_pos)] = p.mean ()
        Nx=Ny =self.Npix
        def neighbors(radius, rowNumber, columnNumber):
                return [[p[i][j] if  i >= 0 and i < Nx and j >= 0 and j < Ny else 0
                for j in range(columnNumber-1-radius, columnNumber+radius)]
                    for i in range(rowNumber-1-radius, rowNumber+radius)]

        neighbours = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                                       for y2 in range(y-1, y+2)
                                       if (-1 < x <= Nx and
                                           -1 < y <= Ny and
                                           (x != x2 or y != y2) and
                                           (0 <= x2 <= Nx) and
                                           (0 <= y2 <= Ny))]

        for iterations  in range(self.niter  ):
            for i,j in zip(  x,y):
                #p[i,j] = np.array([p [k,l]  for k,l in  neighbors(i,j ) ] ).mean()
                p[i,j] = np.array( neighbors(1, i, j )) .mean()

        return p
