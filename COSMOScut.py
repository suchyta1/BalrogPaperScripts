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
    y1 = yLine(slope, point, xfix[0])
    y2 = yLine(slope, point, xfix[1])
    ax.plot( [xfix[0], xfix[1]], [y1, y2], **plotkwargs )


def PlotLines(ax):
    PlotLine(-3.2, [23.25,-3], ax, plotkwargs={'color':'red'})
    PlotLine(-0.25, [22.5,0.35], ax, plotkwargs={'color':'red'})
    #PlotLine(0.35, [23,-2], ax, plotkwargs={'color':'red'})
    #PlotLine(-0.25, [23.5,-1], ax, plotkwargs={'color':'red'})
    PlotLine(0, [22,-1.1], ax, plotkwargs={'color':'red'})





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

    
    fig, axarr = plt.subplots(1,3, figsize=(18,6), tight_layout=True)

    ax = axarr[0]
    ax.scatter(truth['Mapp_i_subaru'], np.log10(truth['halflightradius']), color='green', lw=0, s=1, alpha=1.0, label="Truth (Green)")
    keep = np.random.choice(len(sim), size=len(sim)/1, replace=False)
    sim = sim[keep]
    ax.scatter(sim['mag_i'], np.log10(sim['halflightradius_0_i']), color='blue', lw=0, s=1, alpha=0.35, label=r"Des $\rightarrow$ Truth (Blue)")
    PlotLines(ax)

    ax.legend(loc='lower left')
    ax.set_xlim( [21.2, 25.2]   )
    ax.set_ylim( [-3, 1] )
    #ax.set_yscale('log')
    ax.set_xlabel(r'$i$ [mag]')
    ax.set_ylabel(r'$\log(r_{50})$ [arcsec]')



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


    plt.show()
