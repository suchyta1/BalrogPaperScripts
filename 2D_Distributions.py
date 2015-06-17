#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
import os
import sys
import esutil
import healpy as hp
import kmeans_radec
import slr_zeropoint_shiftmap
import matplotlib
import matplotlib.pyplot as plt
import Utils
import JacknifeSphere as JK



def AddBoris(cat, borismap, ra, dec, killunseen=True, bnest=False, hpkey='hp', qkey='q'):
    nside = hp.npix2nside(len(borismap))
    hpix = Utils.RaDec2Healpix(cat[ra], cat[dec], nside=nside, nest=bnest)
    q = borismap[hpix]

    if killunseen:
        cut = (q!=hp.UNSEEN)
        cat = cat[cut]
        hpix = hpix[cut]
        q = q[cut]

    #cat = recfunctions.append_fields(cat, [hpkey, qkey], data=[hpix, q], usemask=False)
    return cat, hpix, q


def SplitDistribution(des, balrog, band, spliton='nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_FWHM__mean.fits.gz', bnside=4096, bnest=False, ra='alphawin_j2000', dec='deltawin_j2000', hpkey='hp', qkey='q', coordband='i', slr=False, slrfile='slr_zeropoint_shiftmap_v6_splice_cosmos_griz_EQUATORIAL_NSIDE_256_RING.fits'):
    if not slr:
        spliton = spliton.replace('%s',band)
    ra = '%s_%s' %(ra, coordband)
    dec = '%s_%s' %(dec, coordband)

    if slr:
        my_slr_map = slr_zeropoint_shiftmap.SLRZeropointShiftmap(slrfile, fill_periphery=True)
        m = 'mag_auto_%s'%(band)
        m_shift, m_quality = my_slr_map.GetZeropoint(band, des[ra], des[dec], des[m], interpolate=True)
        b_shift, b_quality = my_slr_map.GetZeropoint(band, balrog[ra], balrog[dec], balrog[m], interpolate=True)

        low = np.percentile(m_shift, 25)
        high = np.percentile(m_shift, 75)
        dlow = (m_shift < low)
        blow = (b_shift < low)
        dhigh = (m_shift > high)
        bhigh = (b_shift > high)

        des2 = [des[dlow], des[dhigh]]
        balrog2 = [balrog[blow], balrog[bhigh]]

        des_q2 = [None, None]
        balrog_q2 = [None, None]

        des_hp2 = [None, None]
        balrog_hp2 = [None, None]

        return [des2, des_q2, des_hp2], [balrog2, balrog_q2, balrog_hp2]

    else:

        borismap = Utils.BorisAsMap(spliton, bnside=bnside, nest=bnest)
        des, des_hp, des_q = AddBoris(des, borismap, ra, dec, bnest=bnest, hpkey=hpkey)
        balrog, balrog_hp, balrog_q = AddBoris(balrog, borismap, ra, dec, bnest=bnest, qkey=qkey)



        '''
        low = np.percentile(des_q, 50)
        dlow = (des_q < low)
        blow = (balrog_q < low)

        high = np.percentile(des_q, 50)
        dhigh = (des_q > high)
        bhigh = (balrog_q > high)
        '''

        u = np.unique(des_hp)
        low = np.percentile(borismap[u], 25)
        high = np.percentile(borismap[u], 75)
        dlow = (des_q < low)
        blow = (balrog_q < low)
        dhigh = (des_q > high)
        bhigh = (balrog_q > high)
        
        des2 = [des[dlow], des[dhigh]]
        balrog2 = [balrog[blow], balrog[bhigh]]

        des_q2 = [des_q[dlow], des_q[dhigh]]
        balrog_q2 = [balrog_q[blow], balrog_q[bhigh]]

        des_hp2 = [des_hp[dlow], des_hp[dhigh]]
        balrog_hp2 = [balrog_hp[blow], balrog_hp[bhigh]]

        a = len(des2[0])/float(len(np.unique(des_hp2[0])))
        b = len(des2[1])/float(len(np.unique(des_hp2[1])))
        aa = len(balrog2[0])/float(len(np.unique(balrog_hp2[0])))
        bb = len(balrog2[1])/float(len(np.unique(balrog_hp2[1])))

        return [des2, des_q2, des_hp2], [balrog2, balrog_q2, balrog_hp2]


def H2D(arr, bins, cols, band):
    b1 = arr[0]
    d1 = arr[1]
    b2 = arr[2]
    d2 = arr[3]


    b1_hist, mbins, rbins =  np.histogram2d(b1['%s_%s'%(cols[0],band)],b1['%s_%s'%(cols[1],band)], bins=bins )
    d1_hist, mbins, rbins =  np.histogram2d(d1['%s_%s'%(cols[0],band)],d1['%s_%s'%(cols[1],band)], bins=bins )
    b1_hist = b1_hist / float(len(b1))
    d1_hist = d1_hist / float(len(d1))
    diff1_hist = b1_hist - d1_hist

    b2_hist, mbins, rbins =  np.histogram2d(b2['%s_%s'%(cols[0],band)],b2['%s_%s'%(cols[1],band)], bins=bins )
    d2_hist, mbins, rbins =  np.histogram2d(d2['%s_%s'%(cols[0],band)],d2['%s_%s'%(cols[1],band)], bins=bins )
    b2_hist = b2_hist / float(len(b2))
    d2_hist = d2_hist / float(len(d2))
    diff2_hist = b2_hist - d2_hist

    r = [ [b1_hist, d1_hist, diff1_hist, b2_hist, d2_hist, d2_hist], [mbins,rbins] ]
    return r



def PlotHist2D(data, band, cols, bins, ax1, fig, autocolor=True, vmin=[], vmax=[], vpos=0, normalize=True, hist=None, cmap='jet'):
    cmap = plt.get_cmap(cmap)
    if hist is None:
        hist1, mbins, rbins =  np.histogram2d(data['%s_%s'%(cols[0],band)],data['%s_%s'%(cols[1],band)], bins=bins )
    else:
        hist1 = hist

    if normalize:
        h1 = hist1/ float(len(data))
    else:
        h1 = hist1
    if autocolor:
        cax = ax1.imshow(h1, origin='lower', extent=[bins[1][0],bins[1][-1],bins[0][0],bins[0][-1]], interpolation='nearest', cmap=cmap)
        vmin.append(np.amin(h1))
        vmax.append(np.amax(h1))
    else:
        cax = ax1.imshow(h1, origin='lower', extent=[bins[1][0],bins[1][-1],bins[0][0],bins[0][-1]], interpolation='nearest', vmin=vmin[vpos], vmax=vmax[vpos], cmap=cmap)
    cbar = fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    return h1


def Hist2D(data, band, cols, bins, axes=None, ax1=None, ax2=None, ax3=None, fig=None, vmin=None, vmax=None, autocolor=True, normalize=True):
    if axes is None:
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        ax1 = ax[0]
        ax2 = ax[1]

    if autocolor: 
        vmin = []
        vmax = []

    h1 = PlotHist2D(data[0], band, cols, bins, ax1, fig, autocolor=autocolor, vmin=vmin, vmax=vmax, normalize=normalize, vpos=0)
    h2 = PlotHist2D(data[1], band, cols, bins, ax2, fig, autocolor=autocolor, vmin=vmin, vmax=vmax, normalize=normalize, vpos=1)

    if axes is None:
        fig, ax3 = plt.subplots(1,1, figsize=(8,6))

    h = h1 - h2
    h = PlotHist2D(data[1], band, cols, bins, ax3, fig, autocolor=autocolor, vmin=vmin, vmax=vmax, vpos=2, normalize=False, hist=h)

    return vmin, vmax, h1, h2, h


def Make2DPlots(band='i', outlabel='size-mag', modest=0, slrinvert=False):
    speedup = Utils.GetUsualMasks()
    #speedup = {}
  
    datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    truth, balrog, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True)

    '''
    cut = (balrog['version_i']==2)
    balrog = balrog[cut]
    cut = (des['version_i']==2)
    des = des[cut]
    '''

    balrog = Utils.ApplyThings(balrog, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=False, invertbench=False, posband='i', **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=False, invertbench=False, posband='i', **speedup)


    files = [None,
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_FWHM__mean.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_maglimit__.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_AIRMASS__mean.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYBRITE__mean.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYSIGMA__mean.fits.gz']

    names = ['slr', 
             'fwhm','maglim','airmass','sky-brightness','sky-sigma']

    
    for i in range(len(files[0:1])):
        if names[i]=='slr':
            slr = True
        else:
            slr = False
        #des2, balrog2 = SplitDistribution(des, balrog, band, spliton='nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_FWHM__mean.fits.gz')
        des2, balrog2 = SplitDistribution(des, balrog, band, spliton=files[i], slr=slr)

        mbins = [np.arange(1, 5, 0.1), np.arange(18, 25, 0.1)]
        tbins = [np.arange(0, 2, 0.05), np.arange(18, 25, 0.1)]
        cs = ['flux_radius','mag_auto']


        fig, axarr = plt.subplots(4,3, figsize=(15,12))
        vmin, vmax, desl, desh, diff_des = Hist2D(des2[0], band, ['flux_radius','mag_auto'], mbins, axes=True, ax1=axarr[1,0], ax2=axarr[1,1], ax3=axarr[1,2], fig=fig, autocolor=True, normalize=True)
        vmin, vmax, balrogl, balrogh, diff_balrog = Hist2D(balrog2[0], band, ['flux_radius','mag_auto'], mbins, axes=True, ax1=axarr[0,0], ax2=axarr[0,1], ax3=axarr[0,2], fig=fig, autocolor=False, normalize=True, vmin=vmin, vmax=vmax)
        plt.tight_layout()


        hists, covs, extra, jextra = JK.JackknifeOnSphere( [balrog2[0][0],des2[0][0],balrog2[0][1],des2[0][1]], ['alphawin_j2000_i','alphawin_j2000_i','alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i','deltawin_j2000_i', 'deltawin_j2000_i'], H2D, jargs=[mbins,cs,band], jtype='read', jfile='fullJK-24.txt', varonly=True)
        balrogl, desl, dl, balrogh, desh, dh = hists
        ebl, edl, el, ebh, edb, eh = covs

        dl = np.zeros(desl.shape)
        cut = (desl > 0)
        dl[cut] = (balrogl[cut] - desl[cut])/desl[cut]
        dl = PlotHist2D(dl, band,  ['flux_radius','mag_auto'], mbins, axarr[2,0], fig, autocolor=False, vmin=[-0.5], vmax=[0.5], vpos=0, normalize=False, hist=dl, cmap='bwr')
        #el = np.sqrt( (1-desl) * desl / len(des2[0][0]) + (1-balrogl) * balrogl / len(balrog2[0][0]) )
        #dl = (balrogl - desl)/el
        #dl = PlotHist2D(dl, band,  ['flux_radius','mag_auto'], mbins, axarr[2,0], fig, autocolor=True, normalize=False, hist=dl, cmap='bwr')
        
        ddl = dl / np.sqrt(el)
        ddl = PlotHist2D(ddl, band,  ['flux_radius','mag_auto'], mbins, axarr[2,0], fig, autocolor=True, normalize=False, hist=ddl, cmap='bwr')

        #print len(des2[0][0])
        #print np.amax(desl)
        #print np.amax(el)

        dh = np.zeros(desh.shape)
        cut = (desh > 0)
        dh[cut] = (balrogh[cut] - desh[cut])/desh[cut]

        alpha = np.ones( dh.shape )
        print balrogh * len(balrog2[0][1])
        print np.amax( balrogh * len(balrog2[0][1]) )
        cut = (balrogh * len(balrog2[0][1]) < 100)
        #dh[cut] = 0
        alpha[cut] = 0

        norm = matplotlib.colors.Normalize(-0.5, 0.5)
        carr = plt.get_cmap('bwr')(norm(dh))
        carr[:, :, 3] = alpha
        #dh = np.dstack( [dh, dh, dh, alpha] )
        dh = carr

        dh = PlotHist2D(dh, band,  ['flux_radius','mag_auto'], mbins, axarr[2,1], fig, autocolor=False, vmin=[-0.5], vmax=[0.5], vpos=0, normalize=False, hist=dh, cmap='bwr')

        d = np.zeros(diff_des.shape)
        cut = (diff_des != 0)
        d[cut] = (diff_balrog[cut] - diff_des[cut])/diff_des[cut]
        d = PlotHist2D(d, band,  ['flux_radius','mag_auto'], mbins, axarr[2,2], fig, autocolor=False, vmin=[-1.0], vmax=[1.0], vpos=0, normalize=False, hist=d, cmap='bwr')

        #fig, axarr = plt.subplots(2,3, figsize=(15,8))
        Hist2D(balrog2[0], band, ['halflightradius_0','mag'], tbins, axes=True, ax1=axarr[3,0], ax2=axarr[3,1], ax3=axarr[3,2], fig=fig, autocolor=True, normalize=False)

        #hinput = PlotHist2D(truth, band,  ['halflightradius_0','mag'], [rt_bins,mt_bins], axarr[3,0], fig, autocolor=True, normalize=False)
        plt.tight_layout()

        outdir = os.path.join('Plots', outlabel, band)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%s.png'%(names[i]))
        plt.savefig(outfile)
        #plt.show()


def NHist(arr, cols, bins, band):
    balrogl, mmbins, mrbins =  np.histogram2d(arr['%s_%s'%(cols[0],band)], arr['%s_%s'%(cols[1],band)], bins=bins )
    balrogl = balrogl / float(len(arr))
    return balrogl, len(arr)


def Make2DPlots2(band='i', outlabel='size-mag', modest=0, slrinvert=False, nxticks=None, nyticks=None):
    speedup = Utils.GetUsualMasks()
    #speedup = {}
  
    datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    truth, balrog, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=True)

    '''
    cut = (balrog['version_i']==2)
    balrog = balrog[cut]
    cut = (des['version_i']==2)
    des = des[cut]
    '''

    balrog = Utils.ApplyThings(balrog, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=True, invertbench=False, posband='i', **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=False, colorcut=True, badflag=True, elimask=True, benchmask=True, invertbench=False, posband='i', **speedup)


    files = [#None,
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_FWHM__mean.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_maglimit__.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_AIRMASS__mean.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYBRITE__mean.fits.gz',
             '../BalrogRandoms/nside4096_oversamp4/SVA1_IMAGE_SRC_band_%s_nside4096_oversamp4_SKYSIGMA__mean.fits.gz']

    names = [#'slr', 
             'fwhm','maglim','airmass','sky-brightness','sky-sigma']

    labels = [#'mag',
              r'PSF FWHM [pixel]', r'Mag Lim [mag]', r'Airmass', r'Sky Brightness [ADU]', r'Sky Sigma [ADU]']
            
    
    #for i in range(len(files[0:1])):
    for i in range(len(files)):
        if names[i]=='slr':
            slr = True
        else:
            slr = False
        des2, balrog2 = SplitDistribution(des, balrog, band, spliton=files[i], slr=slr)

        mbins = [np.arange(1, 4.5, 0.1), np.arange(20, 25, 0.1)]
        tbins = [np.arange(0, 2, 0.05), np.arange(18, 25, 0.1)]
        cols = ['flux_radius','mag_auto']


        fig, axarr = plt.subplots(4,3, figsize=(20,16))
        axarr[3,2].axis('off')
        balrogl, nbl = NHist(balrog2[0][0], cols, mbins, band)
        desl, ndl = NHist(des2[0][0], cols, mbins, band)
        balrogh, nbh = NHist(balrog2[0][1], cols, mbins, band)
        desh, ndh = NHist(des2[0][1], cols, mbins, band)

        diff_balrog = balrogl - balrogh
        diff_des = desl - desh
        d = np.zeros(diff_des.shape)
        cut = (diff_des != 0)
        d[cut] = (diff_balrog[cut] - diff_des[cut])/diff_des[cut]
        

        dl = np.zeros(desl.shape)
        cut = (desl > 0)
        dl[cut] = (balrogl[cut] - desl[cut])/desl[cut]

        dh = np.zeros(desh.shape)
        cut = (desh > 0)
        dh[cut] = (balrogh[cut] - desh[cut])/desh[cut]

        ul = UniqueSys(des2[1][0], des2[2][0])
        axarr[3,0].hist(ul, bins=30)
        axarr[3,0].set_xlabel(labels[i])
        axarr[3,0].set_ylabel(r'Number')
        Utils.NTicks(axarr[3,0], nxticks=nxticks, nyticks=nyticks)
        
        uh = UniqueSys(des2[1][1], des2[2][1])
        axarr[3,1].hist(uh, bins=30)
        axarr[3,1].set_xlabel(labels[i])
        axarr[3,1].set_ylabel(r'Number')
        Utils.NTicks(axarr[3,1], nxticks=nxticks, nyticks=nyticks)
    
        usualx = r'\texttt{MAG\_AUTO} $\left[ \mathrm{mag} \right]$'
        usualy = r'\texttt{FLUX\_RADIUS} $\left[ \mathrm{pixel} \right]$'
        usualkwargs = {'xlabel':usualx, 'ylabel':usualy, 'nxticks':nxticks, 'nyticks':nyticks}

        cutl = (desl * ndl < 100)
        iarr, marr, scale = TwoDPlot(desl, cutl, fig, axarr[1,0], mbins, scale=None, cmap='jet', sfmt=True, **usualkwargs)
        iarr, marr, scale = TwoDPlot(balrogl, cutl, fig, axarr[0,0], mbins, scale=scale, cmap='jet', sfmt=True, **usualkwargs)
        iarr, marr, scale = TwoDPlot(dl, cutl, fig, axarr[2,0], mbins, scale=[-0.25,0.25], cmap='bwr', **usualkwargs)
        

        cuth = (desh * ndh < 100)
        iarr, marr, scale = TwoDPlot(desh, cuth, fig, axarr[1,1], mbins, scale=None, cmap='jet', sfmt=True, **usualkwargs)
        iarr, marr, scale = TwoDPlot(balrogh, cuth, fig, axarr[0,1], mbins, scale=scale, cmap='jet', sfmt=True, **usualkwargs)
        iarr, marr, scale = TwoDPlot(dh, cuth, fig, axarr[2,1], mbins, scale=[-0.25,0.25], cmap='bwr', **usualkwargs)


        cutc = (cutl & cuth)
        iarr, marr, scale = TwoDPlot(diff_balrog, cutc, fig, axarr[1,2], mbins, scale=None, cmap='jet', sfmt=True, **usualkwargs)
        iarr, marr, scale = TwoDPlot(diff_des, cutc, fig, axarr[0,2], mbins, scale=scale, cmap='jet', sfmt=True, **usualkwargs)
        iarr, marr, scale = TwoDPlot(d, cutc, fig, axarr[2,2], mbins, scale=[-0.5,0.5], cmap='bwr', **usualkwargs)
        
        
        plt.figtext(0.01, 0.9, r'\textsc{Balrog}', rotation='vertical', fontsize=24)
        plt.figtext(0.01, 0.64, r'DES', rotation='vertical', fontsize=24)
        plt.figtext(0.01, 0.42, r'Difference', rotation='vertical', fontsize=24)

        plt.figtext(0.15, 0.98, r'Percentile $<$ 25', fontsize=24)
        plt.figtext(0.47, 0.98, r'Percentile $>$ 75', fontsize=24)
        plt.figtext(0.78, 0.98, r'Difference', fontsize=24)

        plt.tight_layout(pad=4, h_pad=1.7, w_pad=0.2)
        outdir = os.path.join('Plots', outlabel, band)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%s.pdf'%(names[i]))
        plt.savefig(outfile)
        #plt.show()


def UniqueSys(sys, hpp):
    hh = np.zeros( len(hpp), dtype=[('index',np.int32), ('hp',np.int32)] )
    hh['index'] = np.arange(len(hpp))
    hh['hp'] = hpp
    hh = np.sort(hh, order='hp')
    u = np.unique(hh['hp'])
    ind = np.searchsorted(hh['hp'], u)
    use = hh['index'][ind]
    return sys[use]


def TwoDPlot(arr, cut, fig, ax, bins, scale=None, cmap='jet', xlabel=None, ylabel=None, sfmt=False, nxticks=None, nyticks=None):
    iarr, marr, scale, cmap = ColorMask(arr, cut, scale=scale, cmap=cmap)
    WithMask(iarr, marr, ax, bins, scale, cmap, fig, sfmt=sfmt)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    Utils.NTicks(ax, nxticks=nxticks, nyticks=nyticks)

    return iarr, marr, scale


def WithMask(iarr, marr, ax, mbins, scale, cmap, fig, sfmt=False):
    iax = ax.imshow(iarr, origin='lower', extent=[mbins[1][0],mbins[1][-1],mbins[0][0],mbins[0][-1]], interpolation='nearest', cmap=plt.get_cmap(cmap), vmin=scale[0], vmax=scale[1])
    if sfmt:
        cbar = fig.colorbar(iax, ax=ax, fraction=0.046, pad=0.04, format='%.1e')
    else:
        cbar = fig.colorbar(iax, ax=ax, fraction=0.046, pad=0.04)
    max = ax.imshow(marr, origin='lower', extent=[mbins[1][0],mbins[1][-1],mbins[0][0],mbins[0][-1]], interpolation='nearest', cmap=plt.get_cmap('Greys'))


def ColorMask(arr, cut, scale=None, cmap='jet'):
    ialpha = np.ones( arr.shape )
    malpha = np.zeros( arr.shape )
    ialpha[cut] = 0
    malpha[cut] = 0.6

    if scale is None:
        scale = [np.amin(arr), np.amax(arr)]

    imap = plt.get_cmap(cmap)
    mmap = plt.get_cmap('Greys')
    inorm = matplotlib.colors.Normalize(scale[0], scale[1])
    mnorm = matplotlib.colors.Normalize(0, 0.1)
    iarr = imap(inorm(arr))
    marr = mmap(mnorm(malpha))
    iarr[:, :, 3] = ialpha
    marr[:, :, 3] = malpha
    return iarr, marr, scale, cmap


if __name__=='__main__': 
    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)
    
    Make2DPlots2(band='i', outlabel='size-mag-mask-bench', modest=0, nxticks=6, nyticks=6)

