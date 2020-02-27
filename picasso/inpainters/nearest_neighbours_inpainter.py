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
    """
    Nearest-neighbours inpainting by diffusing the average of the nearest pixel values.

    Inpainting can be performed on  a generic location in the map.
    It is performed in a loop and it  is stopped when the inpainted array
    in two subsequent iterations is essentially the same for
    a given tolerance threshold ``tol``.
    Default tolerance is the one set from numpy.allclose, ``1e-8``.
    """


    def __init__ (self, Npix =128 ,   verbose = False ,  tol=1e-8  ) :

        self.verbose=  verbose
        self.Npix =Npix
        self.tol = tol
        pass


    def setup_input (self, fname_masked  )   :
        """
        Setup the the masked image  to be inpainted and the mask

        **Parameters**

        - ``fname_masked``:{string}
            loading ``.npy`` file of the masked image

        """

        self.X =np.load(fname_masked)
        self.mask = np.int_ ( np.ma.masked_not_equal(self.X,0) .mask )
        pass

    def predict (self   ):
        """
        Inpainting  the map with Nearest-Neighbours .
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
