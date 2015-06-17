#!/usr/bin/env python

import kmeans_radec
import numpy as np
import os
import sys
import healpy as hp
import numpy as np
import esutil
import atpy
import pyfits
import numpy.lib.recfunctions as rec

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import seaborn as sns

import Utils
import JacknifeSphere as JK
import mapfunc



if __name__=='__main__': 

    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)

    
    band = 'i'
    speedup = Utils.GetUsualMasks()
    datadir = os.path.join(os.environ['GLOBALDIR'],'sva1-umatch')
    sim, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True, notruth=True, needonly=False)
    sim = Utils.ApplyThings(sim, band, slr=False, modestcut=None, mag=False, colorcut=False, badflag=False, jbadflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, nozero=False, noduplicate=False, **speedup)
    des = Utils.ApplyThings(des, band, slr=False, modestcut=None, mag=False, colorcut=False, badflag=False, jbadflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, **speedup)

    usnofile = 'usno_blt20_r12.fits.gz'
    usnonest = False

    map = hp.read_map(usnofile, nest=usnonest)
    map = hp.ud_grade(map, 512)
    nside = hp.npix2nside(map.size)
    pix = Utils.RaDec2Healpix(des['alphawin_j2000_i'], des['deltawin_j2000_i'], nside, nest=usnonest)
    upix = np.unique(pix)
    cut = np.in1d(np.arange(map.size), upix)
    map[-cut] = hp.UNSEEN
    uval = map[upix]
    min = -1
    max = np.percentile(uval, 95)

    mapfunc.visualizeHealPixMap(map, nest=usnonest, title="USNO-map", norm=None, vmin=min, vmax=max)

    plt.show()

