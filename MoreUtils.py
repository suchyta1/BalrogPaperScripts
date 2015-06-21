#!/usr/bin/env python

import kmeans_radec
import numpy as np
import os
import sys
import healpy as hp
import numpy as np
import esutil
import numpy.lib.recfunctions as rec
import Utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import seaborn as sns

import Utils
import JacknifeSphere as JK
import mapfunc


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
    mind = np.arange(len(cat))
    cut = np.in1d(mind, mcat)
    return cat[-cut]
    #return cat[mcat]


def UniformRandom(GeoCuts, size=1e6, rakey='ra', deckey='dec'):

    ramin = GeoCuts[0][1][0]  
    ramax = GeoCuts[1][1][0]
    decmin = GeoCuts[2][1][1]
    decmax = GeoCuts[3][1][1]

    ra = np.random.uniform(ramin,ramax, size)

    tmin = np.cos( np.radians(90.0 - decmax) )
    tmax = np.cos( np.radians(90.0 - decmin) )
    theta = np.degrees( np.arccos( np.random.uniform(tmin,tmax, size) ) )
    dec = 90.0 - theta

    uniform = np.zeros(len(ra), dtype=[('%s'%(rakey),np.float64), ('%s'%(deckey),np.float64)])
    uniform['%s'%(rakey)] = ra
    uniform['%s'%(deckey)] = dec
    
    #return ra, dec
    return uniform


def GetCOSMOS23(maxdist=10.0):

    morph = esutil.io.read('cosmos_morph_cassata_1.1.fits')
    morph = rec.rename_fields(morph, {'RA':'ra', 'DEC':'dec'})
    cuts = GetFiducialCuts() 
    cut = XYCuts(morph['MAG_AUTO_ACS'], np.log10(morph['R_HALF']*0.03), cuts)
    morph = morph[cut]

    m1 = np.zeros( (len(morph),2) )
    m1[:,0] = morph['ra']
    m1[:,1] = morph['dec']
    np.savetxt('morph-pre-mask.txt', m1)

    #enrique = np.loadtxt('zCOSMOS-rndnb104Ia20.004-22.525.023.024.0zCOSMOS-Bext3.dat')
    #e = np.zeros( len(enrique), dtype=[('ra',np.float32), ('dec',np.float32)])
    #e['ra'] = enrique[:,2]
    #e['dec'] = enrique[:,1]

    enrique = np.loadtxt('zCOSMOS-mask.dat')
    e = np.zeros( len(enrique), dtype=[('ra',np.float32), ('dec',np.float32)])
    e['ra'] = enrique[:,1]
    e['dec'] = enrique[:,0]


    masked_morph = ApplyMaskRandoms(morph, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)
    GeoCuts = GetGeometryCuts()
    mcut = XYCuts(masked_morph['ra'], masked_morph['dec'], GeoCuts)
    masked_morph = masked_morph[mcut]
    
    m2 = np.zeros( (len(masked_morph),2) )
    m2[:,0] = masked_morph['ra']
    m2[:,1] = masked_morph['dec']
    np.savetxt('morph-post-mask.txt', m2)


    ecut = XYCuts(e['ra'], e['dec'], GeoCuts)
    e = e[ecut]

    u = UniformRandom(GeoCuts)
    u = ApplyMaskRandoms(u, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)
    return masked_morph, e, u

    #return masked_morph, e


def ScaleCovariance(cat1, cat1_cov, cat2, njack=24, nside=4096, nest=False, cat1_ra='alphawin_j2000_i', cat1_dec='deltaawin_j2000_i', cat2_ra='ra', cat2_dec='dec'):
    pix = Utils.RaDec2Healpix(cat1[cat1_ra], cat1[cat1_dec], nside, nest=False)
    upix = np.unique(pix)
    npix = len(upix)
    area = hp.nside2pixarea(nside, degrees=True) * npix
    jarea = area / njack
    print area, jarea


def ContamCorrect(gal_xi, cross_xi, gc=0.0, sc=0.0):
    ss_w = (cross_xi - sc*sc * gal_xi)
    cor = np.power(1 + gc, 2) * (gal_xi - ss_w - np.power(gc,4) / np.power(1+gc, 2) )
    return cor


def GetThing(dir, num, other=0, njack=24, outdir='CorrFiles', band='i', kind='DB'):
    basepath = os.path.join(outdir, dir, band, kind)
    vec = np.loadtxt( os.path.join(basepath, 'vec', '%i.txt'%(num)) )
    cov = np.loadtxt( os.path.join(basepath, 'cov', '%i.txt'%(num)) )
    other = np.loadtxt( os.path.join(basepath, 'other', '%i.txt'%(other)) ) 
    return other, vec, cov


def RecreateContam(dir, njack=24, gc=0, sc=0, outdir='CorrFiles', band='i', kind='DB'):
    basepath = os.path.join(outdir, dir, band, kind)

    vec = os.path.join(basepath, 'vec')
    gg = np.loadtxt(os.path.join(vec,'1.txt'))
    sg = np.loadtxt(os.path.join(vec,'2.txt'))

    sgcov = np.loadtxt(os.path.join(basepath,'cov','2.txt'))

    #cor = ContamCorrect(gg, sg, gc=gc, sc=sc)
    cor = ContaminationCorrection(gg, sg, gc=gc, sc=sc)
    coords = np.loadtxt(os.path.join(basepath, 'other', '0.txt'))

    vec_it = os.path.join(basepath, 'vec_it')
    gg_it = []
    sg_it = []
    cor_it = []
    for i in range(njack):
        g = np.loadtxt(os.path.join(vec,'it','1','%i.txt'%(i)))
        s = np.loadtxt(os.path.join(vec,'it','2','%i.txt'%(i)))
        #c = ContamCorrect(g, s, gc=gc, sc=sc)
        c = ContaminationCorrection(g, s, gc=gc, sc=sc)
        cor_it.append(c)
    cor_it = np.array(cor_it)

    norm = (njack-1) / float(njack)
    csize = len(cor)
    cov = np.zeros( (csize,csize) )
    for i in range(csize):
        for j in range(i, csize):
            cov[i,j] =  np.sum( (cor_it[:,i] - cor[i]) * (cor_it[:,j] - cor[j]) ) * norm

            if i!=j:
                cov[j,i] = cov[i,j]

    return coords, cor, cov


def AshleyCorrection(gg, ss, gc=0.0):
    f2 = np.power(1.0 + gc, 2.0)
    return f2 * (gg - gc*gc*ss - np.power(gc, 4.0)/f2)


def ContaminationCorrection(gg, sg, gc=0.0, sc=0.0):
    f2 = np.power(1.0 + gc, 2.0)
    pre = f2 / (1.0 - gc*sc*f2)
    w = gg - gc*sg - np.power(gc, 4.0)/f2
    return pre*w

def StellarAuto(gg, sg, gc=0.0, sc=0.0):
    f2 = np.power(1.0 + gc, 2.0)
    num = sg - sc*f2*(gg - np.power(gc, 4.0)/f2)
    den = gc * (1.0 - gc*sc*f2)
    return num/den


def MakeHPMap(cat, ra='alphawin_j2000_i', dec='deltawin_j2000_i', nside=512, nest=False, title='Maps/map', min=None, max=None):
    map = np.ones( hp.nside2npix(nside) ) * hp.UNSEEN
    pix = Utils.RaDec2Healpix(cat[ra], cat[dec], nside, nest=nest)
    num = np.bincount(pix, minlength=map.size)
    cut = (num > 0)
    n = num[cut]
    map[cut] = n

    if min is None:
        min = np.percentile(n, 2)
    if max is None:
        max = np.percentile(n, 98)
    mapfunc.visualizeHealPixMap(map, nest=nest, title=title, norm=None, vmin=min, vmax=max)

    #plt.show()
