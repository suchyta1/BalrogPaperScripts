#!/usr/bin/env python

import kmeans_radec
import numpy as np
import os
import sys
import healpy as hp
import numpy as np
import esutil
import numpy.lib.recfunctions as rec

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import seaborn as sns

import Utils
import JacknifeSphere as JK


def RegionCut(xarr, yarr, slope, point, direction='>'):
    if slope is None:
        if direction=='>':
            cut = (xarr > point[0])
        else:
            cut = (xarr < point[0])

    else:
        if direction=='>':
            cut = (yarr > point[1] + slope*(xarr-point[0]))
        else:
            cut = (yarr < point[1] + slope*(xarr-point[0]))
    return cut


def GetFiducialCuts():
    cuts = [ [-3.2, [23.25,-3], '>'],
             [-0.25, [22.5,0.35], '<'],
             #[0, [22,-1.1], '>'],
             #[0, [22,-2.5], '>'],
             [None, [25, None], '<'] ]

    return cuts

def GetGeometryCuts():
    cuts = [ [None, [149.515, None], '>'],
            [None, [150.720, None], '<'],
            [0, [150, 1.658], '>'],
            [0, [150, 2.736], '<'] ]
    return cuts
  

def XYCuts(x, y, cuts):
    cut = np.ones(len(x), dtype=np.bool_)
    for c in cuts:
        cut = (cut & RegionCut(x, y, c[0], c[1], direction=c[2]))
    return cut


def ApplyMaskRandoms(cat, rand, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=5.0):
    maxdist = maxdist / 3600.0
    htm = esutil.htm.HTM()
    mcat, mrand, d = htm.match(cat[cat_ra],cat[cat_dec], rand[rand_ra],rand[rand_dec], maxdist, maxmatch=1)
    return cat[mcat]


def UniformRandom(GeoCuts, size=1e6, rakey='ra', deckey='dec'):
    ramin = GeoCuts[0][1][0],  
    ramax = GeoCuts[1][1][0]
    decmin = GeoCuts[2][1][1]
    decmax = GeoCuts[3][1][1]

    if decmin < 0:
        decmin = 90 - decmin
    if decmax < 0:
        decmax = 90 - decmax

    ra = np.random.uniform(ramin,ramax, size)

    dmin = np.cos(np.radians(decmin))
    dmax = np.cos(np.radians(decmax))
    dec = np.degrees( np.arccos( np.random.uniform(dmin,dmax, size) ) )
    neg = (dec > 90.0)
    dec[neg] = 90.0 - dec[neg]

    uniform = np.zeros(len(ra), dtype=[('%s'%(rakey),np.float64), ('%s'%(deckey),np.float64)])
    uniform['%s'%(rakey)] = ra
    uniform['%s'%(deckey)] = dec
    
    #return ra, dec
    return uniform

def GetCOSMOS23(maxdist=5.0):

    morph = esutil.io.read('cosmos_morph_cassata_1.1.fits')
    morph = rec.rename_fields(morph, {'RA':'ra', 'DEC':'dec'})
    cuts = GetFiducialCuts() 
    cut = XYCuts(morph['MAG_AUTO_ACS'], np.log10(morph['R_HALF']*0.03), cuts)
    morph = morph[cut]

    enrique = np.loadtxt('zCOSMOS-rndnb104Ia20.004-22.525.023.024.0zCOSMOS-Bext3.dat')
    e = np.zeros( len(enrique), dtype=[('ra',np.float32), ('dec',np.float32)])
    e['ra'] = enrique[:,2]
    e['dec'] = enrique[:,1]


    masked_morph = ApplyMaskRandoms(morph, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)
    GeoCuts = GetGeometryCuts()
    mcut = XYCuts(masked_morph['ra'], masked_morph['dec'], GeoCuts)
    masked_morph = masked_morph[mcut]
    ecut = XYCuts(e['ra'], e['dec'], GeoCuts)
    e = e[ecut]

    u = UniformRandom(GeoCuts)
    u = ApplyMaskRandoms(u, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)
    #return masked_morph, e, u

    return masked_morph, e
