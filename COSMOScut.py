#!/usr/bin/env python

import kmeans_radec
import numpy as np
import os
import sys
import healpy as hp
import numpy as np
import esutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import seaborn as sns

import Utils
import MoreUtils
import JacknifeSphere as JK


def yLine(m, point, xfix):
    return m*(xfix-point[0]) + point[1]


def PlotLine(slope, point, ax, xfix=[0,30], plotkwargs={}):
    if slope is None:
        ax.axvline(point[0])
    else:
        y1 = yLine(slope, point, xfix[0])
        y2 = yLine(slope, point, xfix[1])
        ax.plot( [xfix[0], xfix[1]], [y1, y2], **plotkwargs )


def PlotLines(ax, color='black'):
    cuts = MoreUtils.GetFiducialCuts() 
    for cut in cuts:
        PlotLine(cut[0], cut[1], ax, plotkwargs={'color':color})


if __name__=='__main__': 

    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)

    des = esutil.io.read('des-no-v3_2-23-24.fits')
    sim = esutil.io.read('sim-no-v3_2-23-24.fits')
    truth = esutil.io.read('CMC_allband_upsample_with_blanks.fits')

    morph = esutil.io.read('cosmos_morph_cassata_1.1.fits')

    c = (sim['halflightradius_0_i']==0)
    cc = (sim['objtype_i']==3)
    print np.sum(c) / float(len(sim))
    print np.sum(cc) / float(len(sim))

    bins = np.arange(21, 25, 0.05)
    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout=True, num=3)
    plt.hist(sim['mag_i'], bins=bins)
    
    cut = (sim['deltawin_j2000_i'] > -58)
    sim = sim[cut]

    
    s = np.zeros(len(sim), dtype=[('mag_i',np.float32), ('halflightradius_0_i',np.float32)])
    s['mag_i'] = sim['mag_i']
    s['halflightradius_0_i'] = sim['halflightradius_0_i']
    s = np.unique(s)
    print len(sim), len(s)
    sim = s 

    t = np.zeros(len(truth), dtype=[('Mapp_i_subaru',np.float32), ('halflightradius',np.float32)])
    t['Mapp_i_subaru'] = truth['Mapp_i_subaru']
    t['halflightradius'] = truth['halflightradius']
    t = np.unique(t)
    print len(truth), len(np.unique(t))
    truth = t
    
    
    #fig, axarr = plt.subplots(1,3, figsize=(18,6), tight_layout=True, num=1)
    #ax = axarr[0]

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout=True, num=1)
    dummy1, = ax.plot(truth['Mapp_i_subaru'][0], np.log10(truth['halflightradius'][0]), color='green', lw=0, markersize=1, alpha=1.0, linestyle='None', marker='o')
    dummy2, = ax.plot(sim['mag_i'][0], np.log10(sim['halflightradius_0_i'][0]), color='blue', lw=0, markersize=1, alpha=1, linestyle='None', marker='o')

    #fig, axarr = plt.subplots(1,3, figsize=(18,6), tight_layout=True, num=1)
    #ax = axarr[0]

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout=True, num=1)
    ax.scatter(truth['Mapp_i_subaru'], np.log10(truth['halflightradius']), color='green', lw=0, s=1, alpha=1.0)
    keep = np.random.choice(len(sim), size=len(sim)/1, replace=False)
    sim = sim[keep]
    ax.scatter(sim['mag_i'], np.log10(sim['halflightradius_0_i']), color='blue', lw=0, s=1.0, alpha=0.35)
    PlotLines(ax)

    ax.legend((dummy1,dummy2), (r'Full \textsc{Balrog} Truth', r'Observered \textsc{Balrog} Truth'), loc='lower left', markerscale=5, scatterpoints=1, handlelength=1)

    ax.set_xlim( [21.2, 25.2]   )
    ax.set_ylim( [-3, 1] )
    #ax.set_yscale('log')
    ax.set_xlabel(r'$i$ [mag]')
    ax.set_ylabel(r'$\log(r_{50})$ [arcsec]')
    Utils.NTicks(ax, nxticks=6, nyticks=6)


    """
    #fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax = axarr[1]
    ax.scatter(morph['MAG_AUTO_ACS'], np.log10(morph['R_HALF']*0.03), color='green', lw=0, s=1, alpha=1.0, label="COSMOS Morphology")
    PlotLines(ax)
    
    ax.legend(loc='lower left')
    ax.set_xlim( [21.2, 25.2] )
    ax.set_ylim( [-3, 1] )
    ax.set_xlabel(r'$i$ [mag]')
    ax.set_ylabel(r'$\log(r_{50})$ [arcsec]')


    cuts = MoreUtils.GetFiducialCuts() 
    cut = MoreUtils.XYCuts(morph['MAG_AUTO_ACS'], np.log10(morph['R_HALF']*0.03), cuts)

    ax = axarr[2]
    print len(morph)
    morph = morph[cut]
    print len(morph)
    ax.scatter(morph['MAG_AUTO_ACS'], np.log10(morph['R_HALF']*0.03), color='green', lw=0, s=1, alpha=1.0, label="COSMOS Morphology")
    PlotLines(ax)
    
    ax.legend(loc='lower left')
    ax.set_xlim( [21.2, 25.2] )
    ax.set_ylim( [-3, 1] )
    ax.set_xlabel(r'$i$ [mag]')
    ax.set_ylabel(r'$\log(r_{50})$ [arcsec]')

  
    fig, ax = plt.subplots(1,1)

    cut = (sim['z_i'] > 0)
    s = sim[cut]
    m = np.amin(s['z_i'])

    zbins = np.arange(0, 5, 0.03)
    zhist, zbins = np.histogram(s['z_i'], bins=zbins)
    zhist = zhist / float(len(sim))
    c = (zbins[1:] + zbins[:-1]) / 2.0
    c = np.insert(c, 0, 0)
    zhist = np.insert(zhist, 0, np.sum(-cut)/float(len(sim)))
    print np.sum(zhist)

    ax.plot(c, zhist, color='black')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$p(z)$')

    z = np.zeros( (len(zhist),2) )
    z[:,0] = c
    z[:,1] = zhist

    np.savetxt('pz-suchyta.txt', z)
    """

    plt.show()
