#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import numpy.lib.recfunctions as recfunctions
import os
import sys
import esutil
import healpy as hp
import copy

import JacknifeSphere as JK
import Utils


#import seaborn as sns




def PointMap(data, band='i', x='alphawin_j2000', y='deltawin_j2000', ax=None, plotkwargs={}, downfactor=None, downsize=None, title=None, xlim=None, ylim=None):
    x = '{0}_{1}'.format(x,band)
    y = '{0}_{1}'.format(y,band)
  
    if downfactor is not None:
        size = len(data) / downfactor
        keep = np.random.choice(len(data), size=size, replace=False)
    elif downsize is not None:
        keep = np.random.choice(len(data), size=downsize, replace=False)
    else:
        keep = np.ones(len(data), dtype=np._bool)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(data[x][keep],data[y][keep], **plotkwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    return len(data[keep]), ax.get_xlim(), ax.get_ylim()




def BorisNative(file):
    data = esutil.io.read(file)
    return data

def BorisAsMap(file, bnside=4096, nest=False):
    data = esutil.io.read(file)
    pix = data['PIXEL']
    value = data['SIGNAL']
    
    map = np.zeros(hp.nside2npix(bnside))
    map[:] = hp.UNSEEN
    map[pix] = value
    return map


def BySys(d, bins, hpkey='hp', skey='signal'):
    #print 'JK realization'
    data = d[0]
    hps = data[hpkey]
    newvalue = data[skey]

    u = np.unique(hps)
    navg = len(hps) / float(len(u))
    n = np.zeros(len(bins)-1) 
    relerr_n = np.zeros(len(bins)-1)
    relerr_navg2 = 1.0/len(hps)

    c = (bins[1:] + bins[:-1]) / 2.0

    for i in range(len(bins)-1):
        cut = (newvalue > bins[i]) & (newvalue < bins[i+1])
        num = np.sum(cut)
        if num==0:
            continue
        u = np.unique(hps[cut])
        nn = num / float(len(u))
        n[i] = nn / navg
        relerr_n[i] = np.sqrt(relerr_navg2 + 1.0/num) * n[i]

    return [ [n], [c,relerr_n] ]


def GetN(data, bins, hpkey, skey):
    hps = data[hpkey]
    newvalue = data[skey]

    u = np.unique(hps)
    navg = len(hps) / float(len(u))
    n = np.zeros(len(bins)-1) 

    for i in range(len(bins)-1):
        cut = (newvalue > bins[i]) & (newvalue < bins[i+1])
        num = np.sum(cut)
        if num==0:
            continue
        u = np.unique(hps[cut])
        nn = num / float(len(u))
        n[i] = nn / navg
    return n
    

def DiffBySys(d, bins, hpkey='hp', skey='signal'):
    #print 'JK realization'
    balrog = d[0]
    des = d[1]

    bn = GetN(balrog, bins, hpkey, skey)
    dn = GetN(des, bins, hpkey, skey)
    diff = bn - dn
    c = (bins[1:] + bins[:-1]) / 2.0

    return [ [diff], [c] ]


def SeenHPs(map, cat, ra, dec, nside, nest):
    pass

def GetBorisDiff(map, b, d, bins=None, ra='alphawin_j2000_i', dec='deltawin_j2000_i', borisnside=4096, borisnest=False,jfile=None, njack=24):
    pass


def GetBorisPlot(map, boris, orig_cat, ax, ra='alphawin_j2000_i', dec='deltawin_j2000_i', borisnside=4096, borisnest=False, bins=np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]), avg=None, histkwargs={}, jfile=None, njack=24, jax=None, jfig=None, jtitle=None, xtitle=None, hist=True, nxticks=None, nyticks=None, savearea=None):
    cat = np.copy(orig_cat)

    hps = Utils.RaDec2Healpix(cat[ra],cat[dec], nside=borisnside, nest=borisnest)
    '''
    cut = (map[hps] != hp.UNSEEN)
    cat = cat[cut]
    hps = hps[cut]
    '''
    bvalue = map[hps]
    uhps = np.unique(hps)

    nfpix = len(uhps)
    navg = len(hps) / float(nfpix)
    relerr_navg2 = 1.0/len(hps)

    
    if hist is not None:
        m = map[uhps]
        
        cut = (m > bins[0]) & (m < bins[-1])
        mm = m[cut]

        #ax.hist(m, bins=40, **histkwargs)
        hh, bb, something = ax.hist(mm, bins=40, **histkwargs)
        ax.set_ylabel('Number')
        #ax.set_xlabel(r'$Q$')
        if xtitle is not None:
            ax.set_xlabel(xtitle)

        Utils.NTicks(ax, nxticks=nxticks, nyticks=nyticks)
        '''
        if nxticks is not None:
            ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nxticks-1)) )
        if nyticks is not None:
            ax.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nyticks-1)) )
        '''
        
        if savearea is not None:
            arr = np.zeros( (len(hh),2) )
            arr[:,0] = (bb[1:] + bb[:-1]) / 2.0
            arr[:,1] = hh
            outdir = os.path.join(savearea, 'hist')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = os.path.join(outdir, 'hist.txt')
            np.savetxt(outfile, arr)
            outfile = os.path.join(outdir, 'bins.txt')
            np.savetxt(outfile, bb)




    if avg is None:
        avg = np.average(map[uhps])
        #avg = np.average(boris['SIGNAL'])
    #newvalue = bvalue/avg
    newvalue = bvalue

    if bins is None:
        left = np.percentile(newvalue, 2)
        right = np.percentile(newvalue, 98)
        #left = np.amin(newvalue)
        #right = np.amax(newvalue)
        bins = np.linspace(left, right, 10)

    n = np.zeros(len(bins)-1) 
    relerr_n = np.zeros(len(bins)-1)
    v = (bins[1:]+bins[:-1])/2.0
    range_x = np.empty(0)
    range_y = np.empty(0)

    print 'appending'
    cat = recfunctions.append_fields(cat, ['hp', 'signal'], data=[hps, newvalue], usemask=False)
    print 'done appending'

    if jfile is not None:
        hists, covs, extra, jextra = JK.JackknifeOnSphere( [cat], [ra], [dec], BySys, jargs=[bins], jtype='read', jfile=jfile, save=savearea)
        n = hists[0]
        relerr_n = extra[0]
        jcov = covs[0]
        jerr = np.sqrt(np.diag(jcov))

        

        '''
        print 'doing region JK'
        centers = np.loadtxt(jfile)
        km = kmeans_radec.KMeans(centers)
        radec = np.zeros( (len(hps),2) )
        radec[:,0] = cat[ra]
        radec[:,1] = cat[dec]
        index = km.find_nearest(radec)


        for i in range(len(bins)-1):
            cut = (newvalue > bins[i]) & (newvalue < bins[i+1])
            num = np.sum(cut)
            if num==0:
                continue
            u = np.unique(hps[cut])
            nn = num / float(len(u))
            n[i] = nn / navg
            relerr_n[i] = np.sqrt(relerr_navg2 + 1.0/num) * n[i]


        jns = []
        for j in range(njack):
            jcut = -(index==j)
            h = hps[jcut]
            u = np.unique(h)
            nv = newvalue[jcut]
            navg = len(h) / float(len(u))

            jn = np.zeros(len(bins)-1) 

            for i in range(len(bins)-1):
                cut = (nv > bins[i]) & (nv < bins[i+1])
                num = np.sum(cut)
                if num==0:
                    continue
                uu = np.unique(h[cut])
                nn = num / float(len(uu))
                jn[i] = nn / navg
            jns.append(jn)


        jns = np.array(jns)
        b = len(bins)-1
        jcov = np.zeros( (b,b) )

        for i in range(b):
            for j in range(i, b):
                jcov[i,j] =  np.sum( (jns[:,i] - n[i]) * (jns[:,j] - n[j]) ) * float(njack-1)/njack
                if i!=j:
                    jcov[j,i] = jcov[i,j]
        jerr = np.sqrt(np.diag(jcov))
        '''

        if jax is not None:
            plotscale = 1e-4
            nticks = 9
            jp = jcov / plotscale
            jplot = np.arcsinh(jp)

            min = np.amin(jplot)
            max = np.amax(jplot)
            scale = np.amax( np.fabs( [min,max] ) )
            ticks = np.linspace(-scale, scale, num=nticks)
            labels = np.sinh(ticks) * plotscale
            tlabels = []

            for i in range(len(labels)):
                tlabels.append('%.2e'%(labels[i]))

            cax = jax.imshow(jplot, origin='lower', interpolation='nearest', extent=[bins[0], bins[-1], bins[0], bins[-1]], vmin=-scale, vmax=scale, cmap=plt.get_cmap('bwr'))
            cbar = jfig.colorbar(cax, ax=jax, ticks=ticks)
            cbar.ax.set_yticklabels(tlabels)

            jax.set_title(jtitle)
            if xtitle is not None:
                jax.set_xlabel(xtitle)
                jax.set_ylabel(xtitle)



    else:
        jerr = np.zeros(len(bins)-1)
        serr = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            cut = (newvalue > bins[i]) & (newvalue < bins[i+1])
            num = np.sum(cut)
            if num==0:
                continue
            u = np.unique(hps[cut])
            nn = num / float(len(u))
            n[i] = nn / navg
            relerr_n[i] = np.sqrt(relerr_navg2 + 1.0/num) * n[i]

            hc = hps[cut]
            hcb = np.sort(u)
            hcb = np.append(hcb, hcb[-1]+1)
            hist, hcb = np.histogram(hc, bins=hcb)
            nhist = hist/navg
            range_x = np.append(range_x, np.array([v[i]]*len(hist)) )
            range_y = np.append(range_y, nhist)
            #std = np.std(hist) / np.sqrt(len(hist))

            std = np.std(nhist) / np.sqrt(len(hist))
            #std = np.average( [np.percentile(nhist,17), np.percentile(nhist,83)] )
            #std = np.std(nhist)
            serr[i] = std


            jarr = np.zeros(njack)
            rind = np.random.randint(0, high=njack, size=len(u))
            for j in range(njack):
                remove = (rind==j)
                use = u[-remove]
                hpcut = np.in1d(hc, use)
                jnum = np.sum(hpcut) / float(len(use))
                jarr[j] = jnum / navg
            jstd = np.std(jarr) * np.sqrt(njack)
            jerr[i] = jstd

            '''
            for uu in u:
                ucut = (hc==u)
                range_x.append(v[i])
                range_y.append(np.sum(u))
            '''
    #return v,n,relerr_n, range_x, range_y, serr, bins, avg
    return v,n,relerr_n, range_x, range_y, jerr, jcov, bins, avg, cat


def AshleyPlots(file, boris, map, avg, ax):
    '''
    cut = (map > hp.UNSEEN)
    avg = np.average(map[cut])
    #avg = np.average(boris['SIGNAL'])
    '''
    #if file=='DESsv14gbpzvdepth02jackerr.dat':
    if file=='DESsv14gbpzvdepthBLi02jackerr.dat':
        data = np.loadtxt(file)
        val = data[:,1]
        err = data[:,2]
        #x = np.linspace(22.5, 23.56, len(data)) / avg
        x = np.linspace(23.3, 24.2, len(data))
        #x = np.linspace(22.5, 23.56, len(data))
        ax.errorbar(x, val, yerr=err, color='green', label='Ashley')
    elif file=='DESsv14gbpzvseei02jackerr.dat':
        data = np.loadtxt(file)
        val = data[:,1]
        err = data[:,2]
        x = np.linspace(0.81, 1.27, len(data)) * avg
        #x = np.linspace(0.81, 1.27, len(data))
        #x = np.linspace(22.5, 23.56, len(data))
        ax.errorbar(x, val, yerr=err, color='green', label='Ashley')
        pass


def Compare2Boris(sim, des, map, boris, fig, axarr, ind, fig_diff, axarr_diff, bench=None, sample=None, title=None, xlabel=None, bins=np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]), ra='alphawin_j2000_i', dec='deltawin_j2000_i', external=None, jfile=None, njack=24, legend=False, nxticks=None, nyticks=None, savearea=None):
    #fig = plt.figure(figsize=(16,9))
    #gs = gridspec.GridSpec(2,2, width_ratios=[2,1])
    #axrr = np.array([ [plt.subplot(gs[0]),plt.subplot(gs[1])], [plt.subplot(gs[2]),plt.subplot(gs[3])] ])
    
    #fig, axarr = plt.subplots(2, 2, figsize=(16,8))
    
    '''
    ax = axarr[0,0]
    hax = axarr[1,0]
    jax_b = axarr[0,1]
    jax_d = axarr[1,1]
    '''

    ax_diff = axarr_diff[ind][0]
    ax_diff_c = axarr_diff[ind][1]
    ax = axarr[ind][0]
    hax = axarr[ind][1]
    jax_b = None
    jax_d = None

    '''
    nxticks = 6
    nyticks = 6
    '''
    
    if savearea is not None:
        ssave = os.path.join(savearea, 'sim')
        dsave = os.path.join(savearea, 'des')
        diffsave = os.path.join(savearea, 'diff')
    else:
        ssave = None
        dsave = None
    
    nside = hp.npix2nside(len(map))
    vs, s, se, rs_x, rs_y, sstd, scov, bins, avg, sim_cat = GetBorisPlot(map, boris, sim, hax, bins=None, ra=ra, dec=dec, borisnside=nside, histkwargs={'color':'red'}, jfile=jfile, njack=njack, jax=jax_b, jfig=fig, jtitle='Balrog Covariance', xtitle=xlabel, hist=None, nxticks=nxticks, nyticks=nyticks, savearea=ssave)
    BorisPlot(vs, s, sstd, rs_x, rs_y, ax=ax, plotkwargs={'c':'red', 'label':r'\textsc{Balrog}', 'fmt':'o'}, scatterkwargs={'color':'red'}, nxticks=nxticks, nyticks=nyticks, offset=True)

    vd, d, de, rd_x, rd_y, dstd, dcov, bins, avg, des_cat = GetBorisPlot(map, boris, des, hax, bins=bins, avg=avg, ra=ra, dec=dec, borisnside=nside, histkwargs={'color':'blue'}, jfile=jfile, njack=njack, jax=jax_d, jfig=fig, jtitle='DES Covariance', xtitle=xlabel, nxticks=nxticks, nyticks=nyticks, savearea=dsave)
    BorisPlot(vd, d, dstd, rd_x, rd_y, ax=ax, plotkwargs={'c':'blue', 'label':'DES', 'fmt':'o'}, scatterkwargs={'color':'blue'}, nxticks=nxticks, nyticks=nyticks)

    diff_hists, diff_covs, diff_extra, diff_jextrat = JK.JackknifeOnSphere( [sim_cat, des_cat], [ra,ra], [dec,dec], DiffBySys, jargs=[bins], jtype='read', jfile=jfile, save=diffsave)
    diff_hist = diff_hists[0]
    diff_cov = diff_covs[0]
    diff_cent = diff_extra[0]
    diag_cov_vec = np.diag(diff_cov)
    diag_cov = np.diag(diag_cov_vec)

    cc = np.dot(diff_cov.T, diff_cov)
    min_diag = np.amin( np.fabs( np.diag(cc) ) )
    #a2 = 0.01 * min_diag
    a2 = np.amin( np.fabs(cc) ) * 0.001

    #chi2 = Utils.ChiFromZero(diff_hist, diff_cov, rcond=1e-5)
    #chi2 = Utils.TChi2(diff_hist, diff_cov, alpha2=1e-10)
    chi2 = Utils.TChi2(diff_hist, diff_cov, alpha2=a2)
    #print Utils.ChiFromZero(diff_hist, diag_cov)
    #print min_diag
    #print a2
    
    derr = np.sqrt(np.diag(dcov))
    ax_diff.fill_between(diff_cent, -derr, derr, facecolor='yellow', alpha=0.25)
    ax_diff.errorbar(diff_cent, diff_hist, yerr=np.sqrt(np.diag(diff_cov)), fmt='o', color='black')
    if xlabel is not None:
        ax_diff.set_xlabel(xlabel)
    ax_diff.set_ylabel(r'$\Delta \, n / \bar{n}$')
    ax_diff.set_title(r'$\chi^2 / \mathrm{DOF} = %.2f$' %(chi2) )
    Utils.NTicks(ax_diff, nxticks=nxticks, nyticks=nyticks)
    #plt.tight_layout()

    scale = 5e-4
    Utils.ArcsinchImage(diff_cov, ax_diff_c, fig_diff, bins, plotscale=scale, nticks=9, title='Diff Covariance', axislabel=xlabel)
    #cax = ax_diff_c.imshow(diff_cov, origin='lower', interpolation='nearest', extent=[bins[0],bins[-1],bins[0],bins[-1]])
    #cbar = fig_diff.colorbar(cax, ax=ax_diff_c)


    cov = scov + dcov
    icov = np.linalg.pinv(cov)
    diff = d - s
    chi2 = np.dot(diff.T, np.dot(icov,diff)) / len(vs)

    if bench is not None:
        vb, b, be, rb_x, rb_y, bstd, bcov, bins, avg, bench_cat = GetBorisPlot(map, boris, bench, hax, bins=bins, avg=avg, ra=ra, dec=dec, borisnside=nside, histkwargs={'color':'yellow', 'alpha':0.5}, jfile=jfile, njack=njack)
        #vd, d, de, rd_x, rd_y, dstd = GetBorisPlot(map, boris, des, bins=bins, ra=ra, dec=dec)
        #BorisPlot(vd, d, de, rd_x, rd_y, ax=ax, plotkwargs={'c':'blue', 'label':'DES'}, scatterkwargs={'color':'blue', 's':0.5})
        BorisPlot(vb, b, bstd, rb_x, rb_y, ax=ax, plotkwargs={'c':'yellow', 'label':'Bench'}, scatterkwargs={'color':'yellow'})
    
    if sample is not None:
        va, a, ae, ra_x, ra_y, astd, acov, bins, avg, sample_cat = GetBorisPlot(map, boris, sample, hax, bins=bins, avg=avg, ra=ra, dec=dec, borisnside=nside, histkwargs={'color':'cyan', 'alpha':0.5}, jfile=jfile, njack=njack)
        #vd, d, de, rd_x, rd_y, dstd = GetBorisPlot(map, boris, des, bins=bins, ra=ra, dec=dec)
        #BorisPlot(vd, d, de, rd_x, rd_y, ax=ax, plotkwargs={'c':'blue', 'label':'DES'}, scatterkwargs={'color':'blue', 's':0.5})
        BorisPlot(va, a, astd, ra_x, ra_y, ax=ax, plotkwargs={'c':'cyan', 'label':'A. Ross'}, scatterkwargs={'color':'cyan'})

    if external is not None:
        AshleyPlots(external, boris, map, avg, ax)

    min = np.amin( [np.amin(s),np.amin(d)] )
    max = np.amax( [np.amax(s),np.amax(d)] )
    ax.set_ylim([min-0.05,max+0.05])

    if legend:
        ax.legend(loc='best')

    ax.set_ylabel(r'$n/\bar{n}$')
    #ax.set_xlabel(r'$Q/\bar{Q}$')
    if title is not None:
        t = r'%s,  $\chi^2 / \mathrm{DOF} = %.2f$'%(title,chi2)
        #ax.set_title(t)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    #plt.tight_layout()


def BorisPlot(v,n,e, range_x,range_y, ax=None, title=None, plotkwargs={}, scatterkwargs={}, nxticks=None, nyticks=None, offset=False):

    if ax is None:
        fig, ax = plt.subplots()
   
    if offset:
        dv = np.diff(v)
        dv = np.append(dv, dv[-1])
        offset = dv * 0.05
        v = Utils.OffsetX(v, offset)

    ax.errorbar(v, n, yerr=e, **plotkwargs)
    #ax.scatter(range_x, range_y, **scatterkwargs)
    
    if title is not None:
        ax.set_title()

    Utils.NTicks(ax, nxticks=nxticks, nyticks=nyticks)
    '''
    if nxticks is not None:
        ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nxticks-1)) )
    if nyticks is not None:
        ax.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nyticks-1)) )
    '''


def GetBorisFTLA(band):

    files = ['../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_maglimit__.fits.gz'%(band),
             'off',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_FWHM__mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_FWHM_coaddweights_mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYBRITE__mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYBRITE_coaddweights_mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYSIGMA__mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYSIGMA_coaddweights_mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_AIRMASS__mean.fits.gz'%(band),
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_AIRMASS_coaddweights_mean.fits.gz'%(band)]
    
    ts = ['Mag Limit',
          'off',
          'PSF FWHM (mean)',
          'PSF FWHM (coadd weights)',
          'Sky Brightness (mean)',
          'Sky Brightness (coadd weights)',
          'Sky Sigma (mean)',
          'Sky Sigma (coadd weights)',
          'Airmass (mean)',
          'Airmass (coadd weights)']

    labels = ['Mag Limit [mag]',
              'off',
              'PSF FWHM (mean) [pixel]',
              'PSF FWHM (coadd weights) [pixels]',
              'Sky Brightness (mean) [ADU]',
              'Sky Brightness (coadd weights) [ADU]',
              'Sky Sigma (mean) [ADU]',
              'Sky Sigma (coadd weights) [ADU]',
              'Airmass (mean)',
              'Airmass (coadd weights)']

    ashley = [None,
              'off',
              #'DESsv14gbpzvseei02jackerr.dat',
              #'DESsv14gbpzvseei02jackerr.dat',
              ##'DESsv14gbpzvdepth02jackerr.dat',
              #'DESsv14gbpzvdepthBLi02jackerr.dat',
              None,
              None,
              None,
              None,
              None,
              None,
              None,
              None]

    return files, ts, labels, ashley


def SetupFigureGrid2(ts):
    index = 0
    index_diff = 0
    rows = len(ts) / 2

    fig = plt.figure(figsize=(18, 2*len(ts)), tight_layout=True)
    gs = gridspec.GridSpec(rows, 4, width_ratios=[1.3,1, 1.3,1])
    
    fig_diff = plt.figure(figsize=(18, 2*len(ts)), tight_layout=True)
    gs_diff = gridspec.GridSpec(rows, 4)
    
    axarr = []
    axarr_diff = []
    for i in range(len(ts)):
        axarr.append([])
        #axarr_diff.append( fig_diff.add_subplot(gs_diff[index_diff]) )
        #index_diff += 1
        axarr_diff.append( [] )
        for j in range(2):
            axarr[i].append(fig.add_subplot(gs[index]))
            index += 1

            axarr_diff[i].append(fig_diff.add_subplot(gs_diff[index_diff]))
            index_diff += 1
    return fig, axarr, fig_diff, axarr_diff

def SetupFigureGrid(ts):
    index = 0
    index_diff = 0
    rows = len(ts) / 2

    fig = plt.figure(figsize=(18, 2*len(ts)), tight_layout=True)
    gs = gridspec.GridSpec(rows, 4, width_ratios=[1.3,1, 1.3,1])
    
    fig_diff = plt.figure(figsize=(32, 4*len(ts)), tight_layout=True)
    gs_diff = gridspec.GridSpec(rows, 4, width_ratios=[1, 1, 1, 1])
    
    axarr = []
    axarr_diff = []
    for i in range(len(ts)):
        axarr.append([])
        #axarr_diff.append( fig_diff.add_subplot(gs_diff[index_diff]) )
        #index_diff += 1
        axarr_diff.append( [] )
        for j in range(2):
            axarr[i].append(fig.add_subplot(gs[index]))
            index += 1

            axarr_diff[i].append(fig_diff.add_subplot(gs_diff[index_diff]))
            index_diff += 1
    return fig, axarr, fig_diff, axarr_diff


def MakeBorisPlots(sim, des, band, bench=None, sample=None, outdir='boris-bench', jfile=None, njack=24, posband=None, nxticks=None, nyticks=None, savearea=None, fromdir=None):
    if posband is None:
        posband = band
   
    files, ts, labels, ashley = GetBorisFTLA(band)

    outdir = os.path.join(outdir,band)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #ts = ts[0:4]
    fig, axarr, fig_diff, axarr_diff = SetupFigureGrid(ts)

    '''
    gs = gridspec.GridSpec(len(ts), 2, width_ratios=[1.3,1])
    axarr = []
    for i in range(len(ts)):
        axarr.append([])
        for j in range(2):
            axarr[i].append(plt.subplot(gs[2*i+j]))
    '''



    for i in range(len(ts)):
        print ts[i]

        if ts[i]=='off':
            axarr[i][0].axis('off')
            axarr[i][1].axis('off')

            axarr_diff[i][0].axis('off')
            axarr_diff[i][1].axis('off')
            continue

        map = BorisAsMap(files[i])
        #map = hp.ud_grade(map, 2048)
        boris = BorisNative(files[i])

        if i==0:
            legend=True
        else:
            legend = False
        
        if savearea is not None:
            saveto = os.path.join(savearea, labels[i])

        Compare2Boris(sim, des, map, boris, fig, axarr, i, fig_diff, axarr_diff, bench=bench, sample=sample, title=r'%s %s-band'%(ts[i],band), xlabel=labels[i], ra='alphawin_j2000_%s'%(posband), dec='deltawin_j2000_%s'%(posband), external=ashley[i], jfile=jfile, njack=njack, legend=legend, nxticks=nxticks, nyticks=nyticks, savearea=saveto)

        #plt.tight_layout()
        #plt.savefig(os.path.join(outdir,'boris-%s-%s.pdf'%(ts[i],band)))


    fig.savefig(os.path.join(outdir,'boris-all-%s.pdf'%(band)))
    fig_diff.savefig(os.path.join(outdir,'boris-all-%s-diff.pdf'%(band)))
    #plt.show()



def CommonHPs(cat1, cat2, nside=4096, nest=False, ra='alphawin_j2000_i', dec='deltawin_j2000_i'):
    hps1 = Utils.RaDec2Healpix(cat1[ra], cat1[dec], nside, nest=nest)
    hps2 = Utils.RaDec2Healpix(cat2[ra], cat2[dec], nside, nest=nest)

    cut1 = np.in1d(hps1, hps2)
    cut2 = np.in1d(hps2, hps1)

    '''
    fig, axarr = plt.subplots(1,2, figsize=(15,6))
    npoints, xlim, ylim = PointMap(cat1[-cut1], band='i', downfactor=1, plotkwargs={'lw':0, 's':1}, ax=axarr[0], title='Matched')
    npoints, xlim, ylim = PointMap(cat2[-cut2], band='i', downfactor=1, plotkwargs={'lw':0, 's':1}, ax=axarr[1], title='Des')
    plt.show()
    '''

    print np.sum(-cut1), np.sum(-cut2)
    cat1 = cat1[cut1]
    cat2 = cat2[cut2]

    return cat1, cat2


def GetSampleDataBoris(band='i', lower=None, upper=None, invertbench=False, posband='i', modest=0, vers=None):
    speedup = Utils.GetUsualMasks()
    #datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    datadir = os.path.join(os.environ['GLOBALDIR'],'sva1-umatch')
    truth, balrog, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True)

    '''
    cut = (balrog['version_i']==2)
    balrog = balrog[cut]
    cut = (des['version_i']==2)
    des = des[cut]
    '''

    #print len(des), len(balrog)
    #balrog = Utils.ApplyThings(balrog, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=True, invertbench=False, posband='i', **speedup)
    #des = Utils.ApplyThings(des, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=True, invertbench=False, posband='i', **speedup)
    #des, balrog = CommonHPs(des, balrog)

    balrog = Utils.ApplyThings(balrog, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=True, lower=lower, upper=upper, colorcut=True, badflag=True, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False,  nozero=False, noduplicate=False, vers=vers, **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=True, lower=lower, upper=upper, colorcut=True, badflag=True, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, vers=vers, **speedup)

    '''
    ra = 'alphawin_j2000_%s'%(band)
    dec = 'deltawin_j2000_%s'%(band)
    d = np.zeros(len(des), dtype=[(ra,np.float32), (dec,np.float32)])
    d[ra] = des[ra]
    d[dec] = des[dec]
    des = d

    b = np.zeros(len(balrog), dtype=[(ra,np.float32), (dec,np.float32)])
    b[ra] = balrog[ra]
    b[dec] = balrog[dec]
    balrog = b
    '''
    #print len(des), len(balrog)
    #sys.exit()

    return balrog, des


def GetDesSimDiff(basepath):
    paths = ['des','sim','diff']
    stuff = []
    for p in paths:
        path = os.path.join(basepath, p)
        v = GetVecInfo(path)
        stuff.append(v)
    return stuff


def GetVecInfo(path):
    vec = np.loadtxt(os.path.join(path, 'vec', '0.txt'))#[1:-1]
    cov = np.loadtxt(os.path.join(path, 'cov', '0.txt'))#[1:-1, 1:-1]
    vec_coord = np.loadtxt(os.path.join(path, 'other', '0.txt'))#[1:-1]
    return [vec, cov, vec_coord]

def GetDesHist(path):
    p = os.path.join(path, 'des')
    arr = np.loadtxt(os.path.join(p, 'hist', 'hist.txt'))
    c = arr[:,0]
    hist = arr[:,1]
    bins = np.loadtxt(os.path.join(p, 'hist', 'bins.txt'))
    width = np.diff(bins)
    return c, hist, width, bins


def PlotFromDisk(savedir, band, nxticks=None, nyticks=None):
    files, ts, labels, ashley = GetBorisFTLA(band)
    #ts = ts[0:4]
    fig, axarr, fig_diff, axarr_diff = SetupFigureGrid(ts)

    for i in range(len(ts)):

        if ts[i] == 'off':
            axarr[i][0].axis('off')
            axarr[i][1].axis('off')
            axarr_diff[i][0].axis('off')
            axarr_diff[i][1].axis('off')
            continue

        basepath = os.path.join(savedir, labels[i])
        des_arr, sim_arr, diff_arr = GetDesSimDiff(basepath)
        des_hist_c, des_hist, des_hist_width, des_hist_bins = GetDesHist(basepath)

        
        db = (sim_arr[2][1] - sim_arr[2][0]) * 0.05
        sim_arr[2] = Utils.OffsetX(sim_arr[2], offset=db, log=False)

        axarr[i][0].errorbar(des_arr[2], des_arr[0], yerr=np.sqrt(np.diag(des_arr[1])), color='blue', label=r'DES', fmt='o')
        axarr[i][0].errorbar(sim_arr[2], sim_arr[0], yerr=np.sqrt(np.diag(sim_arr[1])), color='red', label=r'\textsc{Balrog}', fmt='o')
        axarr[i][1].bar(des_hist_bins[:-1], des_hist, width=des_hist_width, color='blue')

        scale = 5e-4
        derr = np.sqrt(np.diag(des_arr[1]))
        axarr_diff[i][0].axhline(y=0, ls='--', color='black')
        axarr_diff[i][0].fill_between(diff_arr[2], -derr, derr, facecolor='yellow', alpha=0.25)
        axarr_diff[i][0].errorbar(diff_arr[2], diff_arr[0], yerr=np.sqrt(np.diag(diff_arr[1])), color='black', fmt='o')
        Utils.ArcsinchImage(diff_arr[1], axarr_diff[i][1], fig_diff, des_hist_bins, plotscale=scale, nticks=7, title=r'$\Delta \left( n / \bar{n} \right)$ Covariance', axislabel=None)
        axarr_diff[i][0].set_xlim( [des_hist_bins[0], des_hist_bins[-1]] )

        axarr[i][0].set_xlabel(r'%s'%(labels[i]))
        axarr[i][1].set_xlabel(r'%s'%(labels[i]))
        axarr_diff[i][0].set_xlabel(r'%s'%(labels[i]))
        axarr_diff[i][1].set_xlabel(r'%s'%(labels[i]))

        axarr[i][0].set_ylabel(r'$n / \bar{n}$')
        axarr[i][1].set_ylabel(r'Number')
        axarr_diff[i][0].set_ylabel(r'$\Delta \left( n / \bar{n} \right)$')
        axarr_diff[i][1].set_ylabel(r'%s'%(labels[i]))


        cc = np.dot(diff_arr[1].T, diff_arr[1])
        min_diag = np.amin( np.fabs( np.diag(cc) ) )
        #a2 = 0.01 * min_diag
        #a2 = np.amin( np.fabs(cc) ) * 1e-3
        a2 = 1e-9
        #chi2 = Utils.TChi2(diff_arr[0], diff_arr[1], alpha2=a2)
        chi2 = Utils.ChiFromZero(diff_arr[0], diff_arr[1])
        #chi2 = Utils.ChiFromZero(diff_arr[0], des_arr[1])
        print Utils.ChiFromZero(diff_arr[0], np.diag(np.diag(diff_arr[1])))

        axarr_diff[i][0].set_title(r'$\chi^2 / \mathrm{DOF} = %.2f$'%(chi2))


        if i==0:
            axarr[i][0].legend(loc='best')

        Utils.NTicks(axarr[i][0], nxticks=nxticks, nyticks=nyticks)
        Utils.NTicks(axarr[i][1], nxticks=nxticks, nyticks=nyticks)
        Utils.NTicks(axarr_diff[i][0], nxticks=nxticks, nyticks=nyticks)
        Utils.NTicks(axarr_diff[i][1], nxticks=nxticks, nyticks=nyticks)

    #plt.show()
    outdir = os.path.join('Plots', 'boris-with-diff-benchmark')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig(os.path.join(outdir, 'all-i.pdf'))
    fig_diff.savefig(os.path.join(outdir, 'all-i-diff.pdf'))



if __name__=='__main__': 
    band = 'i'
    posband = 'i'

    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)

    
    '''
    ashley = np.loadtxt('dessv1gal22p5_mo_radecmodwave.dat', usecols=(0,1,2,3), dtype={'names':('alphawin_j2000_%s'%(band), 'deltawin_j2000_%s'%(band), 'modest_%s'%(band), 'wavg_spread_model'), 'formats':('f4','f4','f4','f4')} )
    ashley = ashley[ [ashley['modest_%s'%(band)]==1] ]
    '''

    #savedir = 'BorisSaves/fulltest/%s' %(band)
    #savedir = 'BorisSaves/benchtest/%s' %(band)
    #savedir = 'BorisSaves/fulltest'

    
    
    vers = None
    matched, des = GetSampleDataBoris(band=band, vers=vers, lower=23, upper=24)

    label = 'bench-23-24'
    savedir = 'BorisSaves/%s/%s' %(label,band)
    MakeBorisPlots(matched, des, band, bench=None, sample=None, njack=24, posband=posband, outdir=os.path.join('Plots','BorisPlots', label), nxticks=6, nyticks=6, savearea=savedir, jfile=os.path.join('JK-regions', '24JK-bench-bugfix-23-24') ) 

    #PlotFromDisk(savedir, band, nxticks=6, nyticks=6)

