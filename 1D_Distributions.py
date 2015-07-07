#!/usr/bin/env python

import kmeans_radec
import numpy as np
import os
import healpy as hp
import numpy as np
from scipy import stats
import esutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import seaborn as sns

import Utils
import JacknifeSphere as JK
import MoreUtils


def Histograms(arr, things, b):
    r = [ [], [] ]
    for i in range(len(things)):
        sim = arr[0][things[i]]
        des = arr[1][things[i]]
        bins = b[i]

        cent = (bins[1:] + bins[0:-1]) / 2
        db = np.diff(bins)

        shist, bins = np.histogram(sim, bins=bins)
        dhist, bins = np.histogram(des, bins=bins)

        ss = float(len(sim))
        dd = float(len(des))
        #ss = np.sum(shist)
        #dd = np.sum(dhist)
    
        s = np.log10( shist / (ss * db) )
        d = np.log10( dhist / (dd * db) )
        diff = s - d
        
        r[0].append(np.copy(s))
        r[0].append(np.copy(d))
        r[0].append(np.copy(diff))
        r[1].append(np.copy(cent))
        r[1].append(np.copy(bins))

    return r 


def Histogram(arr, thing, binsize=0.1, bins=None, nums=None):
    sim = arr[0][thing]
    des = arr[1][thing]

    if bins is None:
        bins = np.arange(18, 25.5, binsize)

    cent = (bins[1:] + bins[0:-1]) / 2
    db = np.diff(bins)

    shist, bins = np.histogram(sim, bins=bins)
    dhist, bins = np.histogram(des, bins=bins)


    if nums is None:
        ss = float(len(sim))
        dd = float(len(des))
    else:
        ss = float(nums[0])
        dd = float(nums[1])

    s = np.log10( shist / (ss * db) )
    d = np.log10( dhist / (dd * db) )
    #s = shist / (ss * db) 
    #d = dhist / (dd * db) 

    diff = s - d

    #print np.sqrt(dhist) / (dd * db)

    return [ [s,d, diff], [cent, bins, ss, dd] ]


def Test(band='i', xunit='mag', lu='[', ru=']', ylabel='\mathcal{P}'):
    speedup = Utils.GetUsualMasks()
    #speedup = {}
  
    datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    truth, matched, des, nosim = Utils.GetData2(band=band, dir=datadir, killnosim=True)

    #matched = matched[ matched['modest_i']==0 ]
    #des = des[ des['modest_i']==0 ]

    #bins = np.arange(18, 25.2, 0.2)
    #bins = np.arange(2, 10, 0.2)
    bins = np.arange(0, 0.015, 0.001)
    
    band = 'i'
    matched = Utils.ApplyThings(matched, band, slr=True, slrwrite=None, modestcut=1, mag=False, lower=bins[0], upper=bins[-1], colorcut=True, elimask=True, benchmask=True, invertbench=False, posband='i', **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrwrite=None, modestcut=1, mag=False, lower=bins[0], upper=bins[-1], colorcut=True, elimask=True, benchmask=True, invertbench=False, posband='i', **speedup)
    nums = [len(matched), len(des)]

    #matched = Utils.ApplyThings(matched, band, slr=False, slrwrite=None, modestcut=False, mag=True, lower=bins[0], upper=bins[-1], colorcut=True, elimask=False, benchmask=False, invertbench=False, **speedup)
    #des = Utils.ApplyThings(des, band, slr=False, slrwrite=None, modestcut=False, mag=True, lower=bins[0], upper=bins[-1], colorcut=True, elimask=False, benchmask=False, invertbench=False, **speedup)



    #jfile = JackknifeOnSphere( [nosim], ['ra_i'], ['dec_i'], Dummy, jtype='generate', njack=22, jfile='nosim-JK-22.txt', generateonly=True)
    #print jfile
    #hists, covs, oth, oths = JackknifeOnSphere( [nosim], ['ra_i'], ['dec_i'], Dummy, jtype='read', jfile='nosim-JK-22.txt')
    #print hists, covs, oth, oths


    #jfile = JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['mag_auto_i'], jkwargs={'binsize':0.1}, jtype='generate', jfile='fullJK-24.txt', generateonly=True, njack=24)
    #hists, covs, extra, jextra = JK.JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['mag_auto_%s'%(band)], jkwargs={'nums': None, 'bins': bins, 'binsize':0.2}, jtype='read', jfile='fullJK-24.txt')
    #hists, covs, extra, jextra = JK.JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['flux_radius_%s'%(band)], jkwargs={'nums': None, 'bins': bins, 'binsize':0.2}, jtype='read', jfile='fullJK-24.txt')
    hists, covs, extra, jextra = JK.JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['spreaderr_model_%s'%(band)], jkwargs={'nums': None, 'bins': bins, 'binsize':0.2}, jtype='read', jfile='fullJK-24.txt')
    sim_hist, des_hist, diff_hist = hists
    sim_cov, des_cov, diff_cov = covs
    centers = extra[0]
    bins = extra[1]

    serr = np.sqrt(np.diag(sim_cov))
    derr = np.sqrt(np.diag(des_cov))
    differr = np.sqrt(np.diag(diff_cov))


    #s, se = Utils.TimesPowerLaw(centers, sim_hist, serr, -3.5)
    #d, de = Utils.TimesPowerLaw(centers, des_hist, derr, -3.5)
   
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(2,2, height_ratios=[1,1], width_ratios=[2,1])
    axarr = np.array( [plt.subplot(gs[0]),plt.subplot(gs[2]), plt.subplot(gs[1]), plt.subplot(gs[3])] )
    #fig, ax = plt.subplots(1,1, figsize=(10,6))

    xrange = [bins[0], bins[-1]]


    icov = np.linalg.pinv(diff_cov, rcond=1e-15)
    print diff_hist
    print np.dot(icov,diff_hist)
    chi2 = np.dot(diff_hist.T, np.dot(icov,diff_hist)) / float( len(diff_hist) )
    print chi2

    axarr[0].errorbar(centers, diff_hist, yerr=differr, color='blue', label='Diff', fmt='.')
    axarr[0].set_xlim(xrange)
    scale = 1e-2
    Utils.ArcsinchImage(diff_cov, axarr[2], fig, bins, plotscale=scale, nticks=9, title='Diff Covariance', axislabel=r'\texttt{mag\_auto}')
    scale = 1e2
    Utils.ArcsinchImage(icov, axarr[3], fig, bins, plotscale=scale, nticks=9, title='Diff Inverse Covariance', axislabel=r'\texttt{mag\_auto}')

    '''
    off = diff_hist / differr
    axarr[1].fill_between(xrange, [-1,-1], [1,1], facecolor='yellow', alpha=0.25)
    axarr[1].axhline(y=0)
    axarr[1].scatter(centers, off)
    axarr[1].set_xlim(xrange)
    '''

    axarr[1].errorbar(centers, sim_hist, yerr=serr, color='blue', label='Balrog', fmt='.')
    axarr[1].errorbar(centers, des_hist, yerr=derr, color='red', label='DES', fmt='.')
    axarr[1].set_xlim(xrange)
    axarr[1].legend(loc='best')


    """
    axarr[0].errorbar(centers, sim_hist, yerr=serr, color='blue', label='Balrog', fmt='.')
    axarr[0].errorbar(centers, des_hist, yerr=derr, color='red', label='DES', fmt='.')
    #axarr[0].errorbar(centers, s, yerr=se, color='blue', label='Balrog', fmt='.')
    #axarr[0].errorbar(centers, d, yerr=de, color='red', label='DES', fmt='.')
    axarr[0].set_yscale('log')
    axarr[0].set_xlabel(r'\texttt{mag\_auto} $\left%s \mathrm{%s} \right%s$' %(lu,xunit,ru))
    axarr[0].set_ylabel(r'$%s \left%s \mathrm{%s}^{-1} \right%s$' %(ylabel, lu,xunit,ru))
    #axarr[0].set_ylim( [1e-3, 1e0] )
    axarr[0].set_xlim(xrange)
    axarr[0].legend(loc='best')

    diff = sim_hist - des_hist
    cov_diff = sim_cov + des_cov
    diff_err = np.sqrt(np.diag(cov_diff))
    off = diff / diff_err
    axarr[1].fill_between(xrange, [-1,-1], [1,1], facecolor='yellow', alpha=0.25)
    axarr[1].axhline(y=0)
    axarr[1].scatter(centers, off)
    axarr[1].set_xlim(xrange)
    axarr[1].set_xlabel(r'\texttt{mag\_auto} $\left%s \mathrm{%s} \right%s$' %(lu,xunit,ru))
    axarr[1].set_ylabel(r'$\Delta %s / \sigma_{%s}$' %(ylabel, ylabel))

    sim_icov = np.linalg.pinv(sim_cov, rcond=1e-15)
    des_icov = np.linalg.pinv(des_cov, rcond=1e-15)

    chi2 = Utils.Chi2Disagreement(sim_hist, sim_cov, des_hist, des_cov, rcond=1e-15)
    print chi2
    print Utils.Chi2Disagreement(sim_hist, np.diag(np.diag(sim_cov)), des_hist, np.diag(np.diag(des_cov)), rcond=1e-15)
    axarr[1].text(22, -1.5, r'$\chi^2 / \mathrm{DOF} = %.2f$' %(chi2))
    
    '''
    for i in range(len(bins)-1):
        scut = (matched['mag_auto_i'] > bins[i]) & (matched['mag_auto_i'] < bins[i+1])
        dcut = (des['mag_auto_i'] > bins[i]) & (des['mag_auto_i'] < bins[i+1])
        print '%f - %f'%(bins[i], bins[i+1]), stats.ks_2samp(matched[scut]['mag_auto_i'], des[dcut]['mag_auto_i'])
    '''

    #scale = 1e-1
    scale = 1e5
    #Utils.ArcsinchImage(sim_cov, axarr[2], fig, bins, plotscale=scale, nticks=9, title='Balrog Covariance', axislabel=r'\texttt{mag\_auto}')
    #Utils.ArcsinchImage(des_cov, axarr[3], fig, bins, plotscale=scale, nticks=9, title='DES Covariance', axislabel=r'\texttt{mag\_auto}')
    Utils.ArcsinchImage(sim_icov, axarr[2], fig, bins, plotscale=scale, nticks=9, title='Balrog Covariance', axislabel=r'\texttt{mag\_auto}')
    Utils.ArcsinchImage(des_icov, axarr[3], fig, bins, plotscale=scale, nticks=9, title='DES Covariance', axislabel=r'\texttt{mag\_auto}')
    """
    
    plt.tight_layout()
    plt.show()


def DistsAndDifference(bands=['i'], modest=None, what='mag_auto', whatbins=None, unit=None, xlabel=None, trim=None, nticks=None, stack='diff'):
    #speedup = Utils.GetUsualMasks()
    speedup = {}
  
    datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    truth, matched, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True)

    '''
    cut = (matched['version_i']==2)
    matched = matched[cut]
    cut = (des['version_i']==2)
    des = des[cut]
    '''

    band = 'i'
    matched = Utils.ApplyThings(matched, band, slr=True, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=False, invertbench=False, posband='i', **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=False, invertbench=False, posband='i', **speedup)


    if stack=='band':
        fig = plt.figure(figsize=(10, 4.5*len(bands)))
        gs = gridspec.GridSpec(len(bands),2, width_ratios=[1,1.5])

    elif stack=='diff':
        fig = plt.figure(figsize=(5*len(bands),9))
        gs = gridspec.GridSpec(2,len(bands))

    axarr = []
    for j in range(len(bands)):
        axarr.append( [] )
        for i in range(2):
            if stack=='diff':
                ind = 2*j + i
                axarr[-1].append( plt.subplot(gs[ind]) )
            
            elif stack=='band':
                ind = i*len(bands) + j
                axarr[-1].append( plt.subplot(gs[ind]) )

    xrange = [whatbins[0], whatbins[-1]]
    if trim is None:
        trim = np.zeros(len(bands))

    for i in range(len(bands)):
        end = len(whatbins) - trim[i]

        hists, covs, extra, jextra = JK.JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['%s_%s'%(what,bands[i])], jkwargs={'bins':whatbins[0:end]}, jtype='read', jfile='fullJK-24.txt')
        sim_hist, des_hist, diff_hist = hists
        sim_cov, des_cov, diff_cov = covs
        centers = extra[0]
        bins = extra[1]

        serr = np.sqrt(np.diag(sim_cov))
        derr = np.sqrt(np.diag(des_cov))
        differr = np.sqrt(np.diag(diff_cov))


        #axarr[0].errorbar(centers, diff_hist, yerr=differr, color='blue', label='Diff', fmt='.')
        axarr[i][0].plot(centers, sim_hist, color='red', label=r'\textsc{Balrog}')
        axarr[i][0].plot(centers, des_hist, color='blue', label=r'DES')
        axarr[i][0].set_xlim(xrange)
        axarr[i][0].set_title(r'$%s-\mathrm{band}$'%(bands[i]))
        if i==0:
            axarr[i][0].legend(loc='best')
        if xlabel is not None:
            if unit is not None:
                u = r' $\left[ \mathrm{%s}  \right]$'%(unit)
                uu = r' $\left[ \mathrm{%s}^{-1}  \right]$'%(unit)
                axarr[i][0].set_xlabel(xlabel + u) 
                axarr[i][0].set_ylabel(r'$\log \mathcal{P}$' + uu)
            else: 
                axarr[i][0].set_xlabel(xlabel)
                axarr[i][0].set_ylabel(r'$\log \mathcal{P}$')
        
        axarr[i][1].axhline(y=0, ls='--')
        axarr[i][1].errorbar(centers, diff_hist, yerr=differr, color='black', fmt='.')
        axarr[i][1].set_xlim(xrange)

        if nticks is not None:
            axarr[i][0].yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nticks-1)) )
            axarr[i][1].yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nticks-1)) )

        #axarr[i][1].set_title(r'$%s-\mathrm{band}$'%(bands[i]))
        axarr[i][1].set_ylabel(r'$\Delta \log \mathcal{P}$')
        if xlabel is not None:
            if unit is not None:
                axarr[i][1].set_xlabel(xlabel + r' $\left[ \mathrm{%s}  \right]$'%(unit)) 
            else: 
                axarr[i][1].set_xlabel(xlabel)
    
    plt.tight_layout(h_pad=3.0, w_pad=1.5, pad=2.5)
    #plt.show()
    plt.savefig('Plots/star-mag-i-dist-wide-v1.pdf')
    #plt.savefig('Plots/mag-dist-wide-v1.pdf')



def SetupFigure(stack, bands, basewidth=5.0, baseheight=4.5):
    if stack=='band':
        fig = plt.figure(figsize=(basewidth*2, baseheight*len(bands)))
        gs = gridspec.GridSpec(len(bands),2)

    elif stack=='diff':
        fig = plt.figure(figsize=(basewidth*len(bands), baseheight*2))
        gs = gridspec.GridSpec(2,len(bands))

    axarr = []
    for j in range(len(bands)):
        axarr.append( [] )
        for i in range(2):
            if stack=='band':
                ind = 2*j + i
                axarr[-1].append( plt.subplot(gs[ind]) )
            
            elif stack=='diff':
                ind = i*len(bands) + j
                axarr[-1].append( plt.subplot(gs[ind]) )
    return fig, axarr


def GeneralDistsAndDifference(bands, things, bins, xlabels, units, modest=0, stack='diff', nticks=None, nxticks=None, leg=0, legloc='best', divide=None, jfile='fullJK-24.txt', vers=None, declow=None, dechigh=None):
    band = 'i'
    """
    speedup = Utils.GetUsualMasks()
    #speedup = {}

    #datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    datadir = os.path.join(os.environ['GLOBALDIR'],'sva1-umatch')
    truth, matched, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True)

    #datadir = os.path.join(os.environ['GLOBALDIR'],'COSMOS-v1')
    #matched, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True, notruth=True, needonly=False, noversion=True)



    '''
    cut = (matched['version_i']==2)
    matched = matched[cut]
    c, declow=-58ut = (des['version_i']==2)
    des = des[cut]
    '''


    matched = Utils.ApplyThings(matched, band, slr=True, slrwrite=None, modestcut=modest, mag=True, lower=19, upper=24, colorcut=True, badflag=True, elimask=True, benchmask=False, invertbench=False, vers=vers, posband='i', declow=declow, dechigh=dechigh, **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrwrite=None, modestcut=modest, mag=True, lower=19, upper=24, colorcut=True, badflag=True, elimask=True, benchmask=False, invertbench=False, vers=vers, posband='i', declow=declow, dechigh=dechigh, **speedup)

    matched = Utils.AddColor(matched, bands=['i','z'])
    des = Utils.AddColor(des, bands=['i','z'])

    '''
    if vers is not None:
        cut = (matched['version_i']==vers)
        matched = matched[cut]
        cut = (des['version_i']==vers)
        des = des[cut]
    '''
    """

    des = esutil.io.read('des-gal.fits')
    matched = esutil.io.read('sim-gal.fits')


    fig, axarr = SetupFigure(stack, bands)
    #hists, covs, extra, jextra = JK.JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histograms, jargs=[things,bins], jtype='read', jfile=jfile)

    
    basedir = os.path.join('1D-hists', 'mags', band)
    '''
    save = os.path.join(basedir, 'DD')
    hists, covs, extra, jextra = JK.AltJackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histograms, jargs=[things,bins], jtype='read', jfile=jfile, save=save, itsave=True, )
    '''

    hists = []
    covs = []
    extra = []
    for i in range(4):
        ind = i * 3
        ond = i * 2
        gb_x, gb_y, gb_err = MoreUtils.GetThing('mags', ind+0, other=ond, outdir='1D-hists', band=band, kind='DD')
        gd_x, gd_y, gd_err = MoreUtils.GetThing('mags', ind+1, other=ond, outdir='1D-hists', band=band, kind='DD') 
        gf_x, gf_y, gf_err = MoreUtils.GetThing('mags', ind+2, other=ond, outdir='1D-hists', band=band, kind='DD') 
        hists.append(gb_y)
        hists.append(gd_y)
        hists.append(gf_y)
        covs.append(gb_err)
        covs.append(gd_err)
        covs.append(gf_err)
        extra.append(gb_x)
        extra.append([])


    if divide is None:
        divide = [1.0]*len(things)

    for i in range(len(things)):
        xrange = [bins[i][0]/divide[i], bins[i][-1]/divide[i]]
        hstart = i * 3
        hend = hstart + 3
        pstart = i * 2
        pend = pstart + 2

        sim_hist, des_hist, diff_hist = hists[hstart:hend]
        sim_cov, des_cov, diff_cov = covs[hstart:hend]
        centers, b = extra[pstart:pend]

        serr = np.sqrt(np.diag(sim_cov))
        derr = np.sqrt(np.diag(des_cov))
        differr = np.sqrt(np.diag(diff_cov))

        #axarr[0].errorbar(centers, diff_hist, yerr=differr, color='blue', label='Diff', fmt='.')
        axarr[i][0].plot(centers/divide[i], sim_hist, color='red', label=r'\textsc{Balrog}')
        axarr[i][0].plot(centers/divide[i], des_hist, color='blue', label=r'DES')
        axarr[i][0].set_xlim(xrange)

        if len(bands[i])==1:
            axarr[i][0].set_title(r'$%s-\mathrm{band}$'%(bands[i]))

        unit = units[i]
        xlabel = xlabels[i]
        
        if i==leg:
            axarr[i][0].legend(loc=legloc)

        if xlabel is not None:
            if divide[i]!=1.0:
                xlabel = '%s / %.1e' %(xlabel, divide[i])

            if unit is not None:
                u = r' $\left[ \mathrm{%s}  \right]$'%(unit)
                uu = r' $\left[ \mathrm{%s}^{-1}  \right]$'%(unit)
                axarr[i][0].set_xlabel(xlabel + u) 
                if i==0:
                    axarr[i][0].set_ylabel(r'$\log_{10} p$' + uu)
            else: 
                axarr[i][0].set_xlabel(xlabel)
                axarr[i][0].set_ylabel(r'$\log_{10} p$')
       
        derr = np.sqrt(np.diag(des_cov))
        axarr[i][1].fill_between(centers/divide[i], -derr, derr, facecolor='yellow', alpha=0.25)
        axarr[i][1].axhline(y=0, ls='--')
        axarr[i][1].errorbar(centers/divide[i], diff_hist, yerr=differr, color='black', fmt='.')
        axarr[i][1].set_xlim(xrange)
        #axarr[i][1].set_ylim( [-0.17, 0.05] )

        xfmt = matplotlib.ticker.ScalarFormatter()
        xfmt.set_powerlimits((-1,2))
        axarr[i][0].xaxis.set_major_formatter(xfmt)
        axarr[i][1].xaxis.set_major_formatter(xfmt)

        if nticks is not None:
            axarr[i][0].yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nticks-1)) )
            axarr[i][1].yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nticks-1)) )
        if nxticks is not None:
            axarr[i][0].xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nxticks-1)) )
            axarr[i][1].xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nxticks-1)) )

        #axarr[i][1].set_title(r'$%s-\mathrm{band}$'%(bands[i]))
        axarr[i][1].set_ylabel(r'$\Delta \log_{10} p$')
        if xlabel is not None:
            if unit is not None:
                axarr[i][1].set_xlabel(xlabel + r' $\left[ \mathrm{%s}  \right]$'%(unit)) 
            else: 
                axarr[i][1].set_xlabel(xlabel)
    
    axarr[1][1].set_ylim( [-0.13, 0.06] )
    axarr[2][1].set_ylim( [-0.13, 0.06] )
    plt.tight_layout(h_pad=3.0, w_pad=3.0, pad=2.5)

    plt.show()
    #plt.savefig('Plots/star-mag-i-dist-wide-v1.pdf')
    #plt.savefig('Plots/mag-dist-wide-v1.pdf')




if __name__=='__main__': 
    
    #sns.axes_style(rc=style)
    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)
    
    #DistsAndDifference(bands=['g','r','i','z'], modest=0, what='mag_auto', whatbins=np.arange(18, 28, 0.2), xlabel=r'\texttt{mag\_auto}', unit='mag', trim=[0,7,8,8], nticks=10)
    #DistsAndDifference(bands=['i'], modest=1, what='mag_auto', whatbins=np.arange(18, 25, 0.2), xlabel=r'\texttt{mag\_auto}', unit='mag', trim=[0,0,0,0], nticks=10, stack='band')
   
    #jfile = 'JK-regions/24JK-bench'
    vers = 2
    jfile=os.path.join('JK-regions', '24JK-bench-bugfix-v%i-23-24'%(vers))

    '''
    #bins = [np.arange(-1, 1.4, 0.05), np.arange(1, 6.5, 0.15), np.arange(-0.02, 0.025, 0.001), np.arange(0, 0.011, 0.00025)]
    #GeneralDistsAndDifference(['iz','i','i','i'], ['mag_auto_iz', 'flux_radius_i','spread_model_i', 'spreaderr_model_i'], bins, [r'\texttt{mag\_auto\_i} - \texttt{mag\_auto\_z}', r'\texttt{flux\_radius}', r'\texttt{spread\_model}', r'\texttt{spreaderr\_model}'], ['mag', 'pixel', 'mag', 'mag'], modest=0, stack='diff', nticks=10, nxticks=6, divide=[1.0, 1.0, 1.0e-2, 1.0e-3], jfile=jfile)
    bins = [np.arange(-1, 1.4, 0.05), np.arange(1, 6.5, 0.15), np.arange(1, 6.5, 0.15), np.arange(0, 0.011, 0.00025)]
    GeneralDistsAndDifference(['iz','i','z','i'], ['mag_auto_iz', 'flux_radius_i','flux_radius_z', 'spreaderr_model_i'], bins, [r'\texttt{MAG\_AUTO\_I} - \texttt{MAG\_AUTO\_Z}', r'\texttt{FLUX\_RADIUS}', r'\texttt{FLUX\_RADIUS}', r'\texttt{SPREADERR\_MODEL}'], ['mag', 'pixel', 'pixel', 'mag'], modest=0, stack='diff', nticks=6, nxticks=6, divide=[1.0, 1.0, 1.0, 1.0e-3], jfile=jfile, legloc='lower center', vers=vers)
    '''

    b = np.arange(18, 28, 0.2)
    bins = [b, b[:-7], b[:-8], b[:-8]]
    #b = np.arange(18, 24, 0.2)
    #bins = [b, b, b, b]
    GeneralDistsAndDifference(['g','r','i','z'], ['mag_auto_g', 'mag_auto_r', 'mag_auto_i', 'mag_auto_z'], bins, [r'\texttt{MAG\_AUTO}']*4, ['mag']*4, modest=0, stack='diff', nticks=6, nxticks=6, jfile=jfile, vers=vers)

    '''
    bins = [np.arange(17.8, 25, 0.2)]
    GeneralDistsAndDifference(['i'], ['mag_auto_i'], bins, [r'\texttt{MAG\_AUTO}'], ['mag'], modest=1, stack='band', nticks=6, nxticks=6, jfile=jfile, declow=-58)
    '''
