*******
PICASSO
*******


Python Inpainter for Cosmological and AStrophysical SOurces
===========================================================


This package  provides a suite of inpainting methodologies aimed at
reconstructing holes on images  (128x128 pixels) extracted from a HEALPIX map.

Three inpainting techniques are included in ``PICASSO`` and can be divided into two main groups:

1. diffusive-based methods (Nearest-Neighbours)
2. learning-based methods that rely on training DCNNs to fill the missing pixels with the predictions  learned from a training data-set (Deep-Prior and Generative Adversarial Networks, GAN ).

For further details see `Puglisi et al. (2020)  < >`.

Requirements
############

- `tensorflow`
- `keras`
- `astropy`
- `reproject`
- `mpi4py` (for parallel inpainting)
- `argparse`
- `neuralgym` with `pip install git+https://github.com/JiahuiYu/neuralgym`

Install
#######

.. code-block:: bash

    git clone https://github.com/giuspugl/picasso
    cd picasso
    python setup.py install

Usage
#####

Scripts are provided to the user in order to perform:

- projection from full sky HEALPIX maps to flat thumbnails images `image_stacker   <https://github.com/giuspugl/picasso/blob/master/picasso/image_stacker_mpi.py>`_
- inpainting on GPUs `inpaint_gpu  <https://github.com/giuspugl/picasso/blob/master/picasso/inpaint_gpu.py>`_
- parallel inpainting on multiple processes (with ``mpi4py`` ) `inpaint_mpi <https://github.com/giuspugl/picasso/blob/master/picasso/inpaint_mpi.py>`_
- projection from flat images to HEALPIX  `inpaint_gpu  <https://github.com/giuspugl/picasso/blob/master/picasso/inpaint_gpu.py>`_

Pretrained models
#################

`GAN  training weights <https://drive.google.com/drive/folders/1oZVSj7pa_zL0xWac2IAFYJoflJAKmIjs>`_

Download the model directories  (rename ``checkpoint.txt``  to ``checkpoint`` because google drive automatically add ext after download) 


Support
#######

If you encounter any difficulty in installing and using the code or you think
you found a bug, please `open an issue
<https://github.com/giuspugl/picasso/issues>`_.
