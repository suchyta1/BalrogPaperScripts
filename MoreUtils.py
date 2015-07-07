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

def GetBrightCuts():
    '''
    cuts = [ [None, [21, None], '>'],
             [None, [22, None], '<'] ]
    '''

    '''
    cuts = [ [None, [23, None], '>'],
             [None, [24, None], '<'] ]
    '''

    cuts = [ [None, [22.5, None], '>'],
             [None, [25.0, None], '<'] ]

    '''
    cuts = [ [None, [21.0, None], '>'],
             [None, [21.7, None], '<'],
             [-0.1, [21, 0.25], '<'] ]
    '''
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
    mcat, mrand, d = htm.match(cat[cat_ra],cat[cat_dec], rand[rand_ra],rand[rand_dec], maxdist, maxmatch=-1)
    mind = np.arange(len(cat))
    cut = np.in1d(mind, mcat)
    return cat[-cut]
    #return cat[mcat]

def RectArea():
    cuts = [ [None, [149.4, None], '>'],
            [None, [150.84, None], '<'],
            [0, [150, 1.495], '>'],
            [0, [150, 2.915], '<'] ]
    return cuts


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


def GetEGCuts():
    cuts = [ [None, [149.42, None], '>'],
            [None, [150.82, None], '<'],
            [0, [150, 1.50], '>'],
            [0, [150, 2.91], '<'] ]
    return cuts


def GetCOSMOS21(maxdist=10.0, usemorph=False, jfile=None, hfile=None, usepz=False):
    
    if hfile is not None:
        morph = esutil.io.read(hfile)
        morph = rec.rename_fields(morph, {'RA':'ra', 'DEC':'dec'})

    elif jfile is not None:
        morph = esutil.io.read(jfile)
        print len(morph)

    elif usemorph:
        morph = esutil.io.read('cosmos_morph_cassata_1.1.fits')
        morph = rec.rename_fields(morph, {'RA':'ra', 'DEC':'dec'})
    
    elif usepz:
        morph = esutil.io.read('cosmos_zphot_mag25.fits')
        morph = rec.rename_fields(morph, {'imag':'i_mag', 'rmag':'r_mag'})
        cut = (morph['auto_flag'] > -1)

    else:
        morph = esutil.io.read('cosmos_phot_20060103.fits')
        morph = rec.rename_fields(morph, {'RA':'ra', 'DEC':'dec'})
        #cut = (morph['star']==0) & (morph['auto_flag']==1) & (morph['i_mask']==0)
        #cut = (morph['star']==0) & (morph['auto_flag']==1) & (morph['i_mask']==0) & (morph['blend_mask']==0)
        #cut = (morph['star']==0) & (morph['auto_flag']==1) & (morph['blend_mask']==0) & (morph['i_mask']==0) & (morph['z_mask']==0) & (morph['V_mask']==0) & (morph['B_mask']==0)
        cut = (morph['blend_mask'] == 0) &  (morph['star'] == 0) & (morph['auto_flag'] > -1)
        #cut = (morph['star']==0) & (morph['i_mask']==0)
        morph = morph[cut]

    '''
    fig, ax = plt.subplots(1,1)
    ax.scatter(morph['ra'], morph['dec'], s=0.2, lw=0)
    plt.show()
    sys.exit()
    '''

    '''
    cuts = GetBrightCuts() 

    #cut = XYCuts(morph['MAG_AUTO_ACS'], np.log10(morph['R_HALF']*0.03), cuts)
    cut = XYCuts(morph['i_mag_auto'], morph['i_mag_auto'], cuts)
    morph = morph[cut]
    '''

    enrique = np.loadtxt('zCOSMOS-mask.dat')
    e = np.zeros( len(enrique), dtype=[('ra',np.float32), ('dec',np.float32)])
    e['ra'] = enrique[:,1]
    e['dec'] = enrique[:,0]

    """
    masked_morph = ApplyMaskRandoms(morph, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)

    if jfile is not None:
        GeoCuts = GetEGCuts()
        mcut = XYCuts(masked_morph['ra'], masked_morph['dec'], GeoCuts)
        masked_morph = masked_morph[mcut]
        r = RectArea()
        u = UniformRandom(r)
        u = ApplyMaskRandoms(u, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)

    elif usemorph:
        GeoCuts = GetGeometryCuts()
        mcut = XYCuts(masked_morph['ra'], masked_morph['dec'], GeoCuts)
        masked_morph = masked_morph[mcut]
        ecut = XYCuts(e['ra'], e['dec'], GeoCuts)
        e = e[ecut]
        u = UniformRandom(GeoCuts)
        u = ApplyMaskRandoms(u, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)
    else:
        r = RectArea()
        u = UniformRandom(r)
        u = ApplyMaskRandoms(u, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)

    return masked_morph, e, u

    #return masked_morph, e
    """

    if hfile is not None:
        GeoCuts = GetEGCuts()
    elif jfile is not None:
        GeoCuts = GetEGCuts()
    elif usemorph:
        GeoCuts = GetGeometryCuts()
    else:
        GeoCuts = GetEGCuts()
    
    mcut = XYCuts(morph['ra'], morph['dec'], GeoCuts)
    morph = morph[mcut]
    u = UniformRandom(GeoCuts)
    u = ApplyMaskRandoms(u, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)
    morph = ApplyMaskRandoms(morph, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=maxdist)

    return morph, e, u




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


def GetVec(n, d1, dir, band, kind, other, njack):
    basepath = os.path.join(dir, d1, band, kind)
    vec = os.path.join(basepath, 'vec')
    o = os.path.join(basepath, 'other')

    v = np.loadtxt(os.path.join(vec,'%i.txt'%(n)))
    oth = np.loadtxt(os.path.join(o,'%i.txt'%(other)))
    v_it = []
    for i in range(njack):
        vv = np.loadtxt(os.path.join(vec,'it','%i'%(n),'%i.txt'%(i)))
        v_it.append(vv)
    v_it = np.array(v_it)

    return [v, v_it, oth]


def JKCov(full, it, njack):
    norm = (njack-1) / float(njack)
    csize = len(full)
    cov = np.zeros( (csize,csize) )
    for i in range(csize):
        for j in range(i, csize):
            cov[i,j] =  np.sum( (it[:,i] - full[i]) * (it[:,j] - full[j]) ) * norm

            if i!=j:
                cov[j,i] = cov[i,j]
    return cov


def GetDifference(d1, d2, num=0, njack=24, ddir='CorrFiles', band='i', kind='DB', other=0):
    v1, v1_it, o1 = GetVec(num, d1, ddir, band, kind, other, njack)
    v2, v2_it, o2 = GetVec(num, d2, ddir, band, kind, other, njack)

    v_diff = v2 - v1
    v_it_diff = v2_it - v1_it
    cov = JKCov(v_diff, v_it_diff, njack)
    return v_diff, cov, o1


def RecreateContam(dir, njack=24, gc=0, sc=0, outdir='CorrFiles', band='i', kind='DB'):
    basepath = os.path.join(outdir, dir, band, kind)

    vec = os.path.join(basepath, 'vec')
    gg = np.loadtxt(os.path.join(vec,'1.txt'))
    sg = np.loadtxt(os.path.join(vec,'2.txt'))
    ss = np.loadtxt(os.path.join(vec, '4.txt'))

    sgcov = np.loadtxt(os.path.join(basepath,'cov','2.txt'))

    #cor = ContamCorrect(gg, sg, gc=gc, sc=sc)
    #cor = ContaminationCorrection(gg, sg, gc=gc, sc=sc)
    cor = AshleyCorrection2(gg, sg, gc=gc, sc=sc)

    coords = np.loadtxt(os.path.join(basepath, 'other', '0.txt'))

    vec_it = os.path.join(basepath, 'vec_it')
    gg_it = []
    sg_it = []
    cor_it = []
    for i in range(njack):
        g = np.loadtxt(os.path.join(vec,'it','1','%i.txt'%(i)))
        s = np.loadtxt(os.path.join(vec,'it','2','%i.txt'%(i)))
        ss = np.loadtxt(os.path.join(vec,'it','4','%i.txt'%(i)))
        #c = ContamCorrect(g, s, gc=gc, sc=sc)
        #c = ContaminationCorrection(g, s, gc=gc, sc=sc)
        c = AshleyCorrection2(g, ss, gc=gc, sc=sc)
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

def AshleyCorrection2(gg, ss, gc=0.0, sc=0.0):
    f2 = np.power(1.0 + gc, 2.0)
    g2 = np.power(1.0 + sc, 2.0)
    return f2 * (gg - g2*gc*gc*ss - np.power(gc, 4.0)/f2)


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


def Selection(band='i', gmag=False, glower=None, gupper=None, smag=False, slower=None, supper=None, declow=None, dechigh=None, vers=None, invertvers=False, killnosim=True):

    print 'getting masks'
    speedup = Utils.GetUsualMasks()
    datadir = os.path.join(os.environ['GLOBALDIR'],'sva1-umatch')

    print 'reading data'
    sim, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=killnosim, notruth=True, needonly=False)


    print 'making cuts'

    #print len(sim)
    sim = Utils.ApplyThings(sim, band, slr=True, slrinvert=False, slrwrite=None, modestcut=-1, mag=None, colorcut=True, badflag=True, jbadflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, declow=declow, dechigh=dechigh, nozero=False, noduplicate=False, vers=vers, invertvers=invertvers, **speedup)
    #print len(sim)

    #print len(des)
    des = Utils.ApplyThings(des, band, slr=True, slrinvert=False, slrwrite=None, modestcut=-1, mag=None, colorcut=True, badflag=True, jbadflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, declow=declow, dechigh=dechigh, vers=vers, invertvers=invertvers, **speedup)
    #print len(des)

    sim_gal = np.copy(sim)
    sim_gal = Utils.ApplyThings(sim_gal, band, slr=False, slrinvert=False, slrwrite=None, modestcut=0, mag=gmag, lower=glower, upper=gupper, colorcut=False, badflag=False, jbadflag=False, elimask=False, benchmask=False, invertbench=False, posband='i', posonly=False, declow=None, dechigh=None, nozero=False, noduplicate=False, vers=None, invertvers=False, **speedup)

    des_gal = np.copy(des)
    des_gal = Utils.ApplyThings(des_gal, band, slr=False, slrinvert=False, slrwrite=None, modestcut=0, mag=gmag, lower=glower, upper=gupper, colorcut=False, badflag=False, jbadflag=False, elimask=False, benchmask=False, invertbench=False, posband='i', posonly=False, declow=None, dechigh=None, vers=None, invertvers=False, **speedup)

    sim_star = np.copy(sim)
    sim_star = Utils.ApplyThings(sim_star, band, slr=False, slrinvert=False, slrwrite=None, modestcut=1, mag=smag, lower=slower, upper=supper, colorcut=False, badflag=False, jbadflag=False, elimask=False, benchmask=False, invertbench=False, posband='i', posonly=False, declow=None, dechigh=None, nozero=False, noduplicate=False, vers=None, invertvers=False, **speedup)

    des_star = np.copy(des)
    des_star = Utils.ApplyThings(des_star, band, slr=False, slrinvert=False, slrwrite=None, modestcut=1, mag=smag, lower=slower, upper=supper, colorcut=False, badflag=False, jbadflag=False, elimask=False, benchmask=False, invertbench=False, posband='i', posonly=False, declow=None, dechigh=None, vers=None, invertvers=False, **speedup)


    obj = 'objtype_%s'%(band)
    mod = 'modest_i'
    star_truth = sim[ sim[obj]==3 ]
    gal_truth = sim[ sim[obj]==1 ]


    f_star2gal = np.sum( star_truth[mod]==0 ) / float(len(star_truth))
    f_star2star = np.sum( star_truth[mod]==1 ) / float(len(star_truth))
    f_gal2gal = np.sum( gal_truth[mod]==0 ) / float(len(gal_truth))
    f_gal2star = np.sum( gal_truth[mod]==1 ) / float(len(gal_truth))

    print f_star2gal, f_star2star, f_star2gal + f_star2star
    print f_gal2gal, f_gal2star, f_gal2gal + f_gal2star
    sys.exit()

    gc = np.sum(sim_gal[obj]==3) / float(len(sim_gal))
    gcontam = float(len(sim_gal))/len(des_gal) * float(len(des_star))/len(sim_star) * gc
    sc = np.sum(sim_star[obj]==1) / float(len(sim_star))
    scontam = float(len(sim_star))/len(des_star) * float(len(des_gal))/len(sim_gal) * sc

    print float(len(des_star))/len(sim_star) * gc
    print sc
    print gcontam, scontam

    #uniform = Utils.ApplyThings(uniform, band, slr=False, slrinvert=False, slrwrite=None, modestcut=None, mag=False, colorcut=False, badflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', ra='ra', dec='dec', declow=declow, dechigh=dechigh, **speedup)
    #keep = np.random.choice(len(uniform), size=10*len(des), replace=False)
    #uniform = uniform[keep]

    ra = 'alphawin_j2000_%s'%(band)
    dec = 'deltawin_j2000_%s'%(band)
    ura = 'ra_%s'%(band)
    udec = 'dec_%s'%(band)

    print 'total S, D:', len(sim), len(des)
    print 'gal S, D', len(sim_gal), len(des_gal)
    print 'star S, D', len(sim_star), len(des_star)


def CapError(y, yerr, small=1e-10):
    diff = y - yerr
    cut = (diff < 0)
    yerr[cut] = y[cut] - small
    return yerr


def ThingPlot(label, what, num, kind, ax, plotkwargs={}, band='i', dir='CorrFiles', other=0, xoffset=None, logxoffset=True, caperr=False, sample=1, start=0, tcut=False, tval=10.0, pscale=None, fill=False):
    f_theta, f_gg, f_ggcov = GetThing(label, num, other=other, outdir=dir, band=band, kind=kind)
   
    if tcut:
        cut = (f_theta > 10.0 / 3600.0)
        ind = np.arange(len(f_theta))
        f_theta = f_theta[cut]
        f_gg = f_gg[cut]

        index = ind[cut]
        min = np.amin(index)
        max = np.amax(index) + 1
        f_ggcov = f_ggcov[min:max, min:max]

    
    if xoffset is not None:
        f_theta = Utils.OffsetX(f_theta, offset=xoffset, log=logxoffset)

    e =  np.sqrt(np.diag(f_ggcov))
    if caperr:
        e = CapError(f_gg, e)

    if pscale is not None:
        f_gg = np.power(f_theta, pscale) * f_gg
        e = np.power(f_theta, pscale) * e

    if fill:
        l = f_gg[start::sample] - e[start::sample]
        u = f_gg[start::sample] + e[start::sample]
        ax.fill_between(f_theta[start::sample], l, u, label='%s'%(what), **plotkwargs)
        p = plt.Rectangle((0,0), 0,0,  **plotkwargs)
    else:
        p = ax.errorbar(f_theta[start::sample], f_gg[start::sample], e[start::sample], label='%s'%(what), **plotkwargs)

    return [p, what]
