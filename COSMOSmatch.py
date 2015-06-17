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
import MoreUtils
import JacknifeSphere as JK


def PlotCompleteness(phot, mmorph, clarie):
    bins = np.arange(18, 26, 0.1)
    c = (bins[1:] + bins[:-1]) / 2.0

    fig, ax = plt.subplots(1,3, figsize=(15,6), tight_layout=True)
    #mhist, mbins, _ = ax[0].hist(mmorph['i_mag_auto'], bins=bins,color='green', label='COSMOS Morphology')
    mhist, mbins, _ = ax[0].hist(mmorph['MAG_AUTO_ACS'], bins=bins,color='green', label='COSMOS Morphology')
    phist, pbins, _ = ax[0].hist(phot['i_mag_auto'], bins=bins, color='red', label='COSMOS Photometry')
    chist, cbins, _ = ax[0].hist(claire['Mapp_i_subaru'], bins=bins, color='blue', label='Claire')
    ax[1].plot(c, np.float32(mhist)/np.float32(phist), color='green')
    ax[2].plot(c, np.float32(chist)/np.float32(phist), color='blue')

    ax[0].legend(loc='best')
    ax[0].set_xlabel(r'\texttt{MAG\_AUTO\_I}')
    ax[1].set_xlabel(r'\texttt{MAG\_AUTO\_I}')
    ax[2].set_xlabel(r'\texttt{MAG\_AUTO\_I}')
    
    ax[1].set_ylabel('M/P')
    ax[2].set_ylabel('C/P')

    ax[1].set_ylim( [0,1.1] )
    ax[2].set_ylim( [0,0.4] )


def PlotPoints(phot, morph, claire):
    fig, ax = plt.subplots(1,3, figsize=(20,6), tight_layout=True)
    npoints, xlim, ylim = Utils.PointMap(phot, band=None, x='RA', y='DEC', plotkwargs={'lw':0, 's':0.2}, ax=ax[0])
    npoints, xlim, ylim = Utils.PointMap(morph, band=None, x='ra', y='dec', plotkwargs={'lw':0, 's':0.2}, ax=ax[1])
    npoints, xlim, ylim = Utils.PointMap(claire, band=None, x='ALPHA_J2000', y='DELTA_J2000', plotkwargs={'lw':0, 's':0.2}, ax=ax[2])

def PlotPoints2(phot, morph, enrique):
    fig, ax = plt.subplots(1,3, figsize=(20,6), tight_layout=True)
    npoints, xlim, ylim = Utils.PointMap(phot, band=None, x='RA', y='DEC', plotkwargs={'lw':0, 's':0.2}, ax=ax[0])
    npoints, xlim, ylim = Utils.PointMap(morph, band=None, x='ra', y='dec', plotkwargs={'lw':0, 's':1.0}, ax=ax[1])
    npoints, xlim, ylim = Utils.PointMap(enrique, band=None, x='ra', y='dec', plotkwargs={'lw':0, 's':0.2}, ax=ax[2])

    ax[0].set_title('COSMOS Photometry')
    ax[1].set_title('COSMOS Morphology')
    ax[2].set_title('Enrique Randoms')
    
    ax[0].set_xlabel('RA [deg]')
    ax[1].set_xlabel('RA [deg]')
    ax[2].set_xlabel('RA [deg]')
    ax[0].set_ylabel('DEC [deg]')
    ax[1].set_ylabel('DEC [deg]')
    ax[2].set_ylabel('DEC [deg]')


def PlotPointsArr(*args, **kwargs):
    size = len(args)
   
    needed = ['x', 'y', 'title', 'xl', 'yl', 's']
    extra = {}
    for n in needed:
        if n not in kwargs:
            kwargs[n] = None
            extra[n] = []
        else:
            extra[n] = kwargs[n]
    
    for i in range(size):
        if kwargs['x'] is None:
            extra['x'].append('ra')
        if kwargs['y'] is None:
            extra['y'].append('dec')
        if kwargs['title'] is None:
            extra['title'].append('')
        if kwargs['xl'] is None:
            extra['xl'].append('RA [deg]')
        if kwargs['yl'] is None:
            extra['yl'].append('DEC [deg]')
        if kwargs['s'] is None:
            extra['s'].append(0.2)


    fig, ax = plt.subplots(1,size, figsize=(6*size,6), tight_layout=True)
    for i in range(size):
        npoints, xlim, ylim = Utils.PointMap(args[i], band=None, x=extra['x'][i], y=extra['y'][i], plotkwargs={'lw':0, 's':extra['s'][i]}, ax=ax[i])
        ax[i].set_title(extra['title'][i])
        ax[i].set_xlabel(extra['xl'][i])
        ax[i].set_xlabel(extra['yl'][i])





if __name__=='__main__': 

    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)

   
    """

    '''
    #morph = atpy.Table('cosmos_morphology_2005.tbl').data
    #esutil.io.write('cosmos_morphology_2005.fits', morph, clobber=True)
    morph = esutil.io.read('cosmos_morphology_2005.fits')
    '''
    #morph = atpy.Table('cosmos_morph_cassata_1.1.tbl').data
    #esutil.io.write('cosmos_morph_cassata_1.1.fits', morph, clobber=True)
    morph = esutil.io.read('cosmos_morph_cassata_1.1.fits')

    #phot = atpy.Table('cosmos_phot_20060103.tbl.gz').data
    #esutil.io.write('cosmos_phot_20060103.fits', phot, clobber=True)
    phot = esutil.io.read('cosmos_phot_20060103.fits')

    claire = pyfits.open('/n/des/suchyta.1/des/magnification/current_balrog/Balrog/CosmosSersicFits/CMC081211_all_goodRAW00000.99988.fits')[1].data

    enrique = np.loadtxt('zCOSMOS-rndnb104Ia20.004-22.525.023.024.0zCOSMOS-Bext3.dat')
    e = np.zeros( len(enrique), dtype=[('ra',np.float32), ('dec',np.float32)])
    e['ra'] = enrique[:,2]
    e['dec'] = enrique[:,1]

    '''
    print phot['i_star']
    print phot['auto_flag']
    print phot['i_mask']
    '''

    cut = (phot['star']==0) & (phot['auto_flag']==1) & (phot['i_mask']==0)
    phot = phot[cut]
    #phot = rec.rename_fields(phot, {'ID':'oid'})
    #mmorph = rec.join_by('oid', morph, phot, usemask=False)
    
    morph = rec.rename_fields(morph, {'RA':'ra', 'DEC':'dec'})
    #mmorph = rec.join_by('ID', morph, phot, usemask=False)
    mmorph = morph
    #print np.sum( np.in1d(morph['ID'],phot['ID']) )
    #print np.amax(morph['ID']), np.amax(phot['ID'])

    cuts = MoreUtils.GetFiducialCuts() 
    cut = MoreUtils.XYCuts(mmorph['MAG_AUTO_ACS'], np.log10(mmorph['R_HALF']*0.03), cuts)
    print len(mmorph)
    mmorph = mmorph[cut]
    print len(mmorph)

    #PlotCompleteness(phot, mmorph, claire)
    #PlotPoints(phot, mmorph, claire)
    #PlotPoints2(phot, mmorph, e)


    masked_morph = MoreUtils.ApplyMaskRandoms(mmorph, e, cat_ra='ra', cat_dec='dec', rand_ra='ra', rand_dec='dec', maxdist=5.0)
    print len(masked_morph)
    PlotPoints2(phot, masked_morph, e)


    GeoCuts = MoreUtils.GetGeometryCuts()
    mcut = MoreUtils.XYCuts(masked_morph['ra'], masked_morph['dec'], GeoCuts)
    masked_morph = masked_morph[mcut]
    print len(masked_morph)

    ecut = MoreUtils.XYCuts(e['ra'], e['dec'], GeoCuts)
    e = e[ecut]


    #PlotPoints2(phot, masked_morph, e)
    """
    

    #masked_morph, e, u  = MoreUtils.GetCOSMOS23()
    #PlotPointsArr(masked_morph, e, u, title=['COSOMS Morphology', 'Enrique Randoms', 'Randoms'], s=[0.5,0.2, 0.2])
    #print len(masked_morph), len(e), len(u)

    masked_morph, e  = MoreUtils.GetCOSMOS23()
    PlotPointsArr(masked_morph, e, title=['COSOMS Morphology', 'Enrique Randoms'], s=[0.5,0.2])
    print len(masked_morph), len(e)

    plt.show()

