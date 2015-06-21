#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
import os
import sys
import esutil
import healpy as hp
import treecorr
import kmeans_radec
import copy
from scipy.optimize import minimize, leastsq
import emcee
import slr_zeropoint_shiftmap
from matplotlib import gridspec
import time
import json
#import seaborn as sns

import matplotlib.pyplot as plt

import Utils
import JacknifeSphere as JK
import PlottingStuff
import MoreUtils



def JKCorr(arrs, datara=None, datadec=None, randomra=None, randomdec=None, corrconfig=None, cross=False):
    if corrconfig is None:
        print 'no corrconfig given, using default'
        corrconfig = {'sep_units': 'arcmin',
                      'min_sep': 0.1,
                      'max_sep': 600.0,
                      'nbins': 40,
                      'bin_slop': 0.25}

    if not cross:
        data = arrs[0]
        random = arrs[1]

        DataCat = treecorr.Catalog(ra=data[datara], dec=data[datadec], ra_units='degrees', dec_units='degrees')
        RandCat = treecorr.Catalog(ra=random[randomra], dec=random[randomdec], ra_units='degrees', dec_units='degrees')
        dd = treecorr.NNCorrelation(**corrconfig)
        dr = treecorr.NNCorrelation(**corrconfig)
        rr = treecorr.NNCorrelation(**corrconfig)
        
        dd.process(DataCat)
        dr.process(DataCat,RandCat)
        rr.process(RandCat)
        xi, varxi = dd.calculateXi(rr, dr)

    else:
        data1 = arrs[0]
        random1 = arrs[1]
        data2 = arrs[2]
        random2 = arrs[3]

        DataCat1 = treecorr.Catalog(ra=data1[datara], dec=data1[datadec], ra_units='degrees', dec_units='degrees')
        RandCat1 = treecorr.Catalog(ra=random1[randomra], dec=random1[randomdec], ra_units='degrees', dec_units='degrees')
        DataCat2 = treecorr.Catalog(ra=data2[datara], dec=data2[datadec], ra_units='degrees', dec_units='degrees')
        RandCat2 = treecorr.Catalog(ra=random2[randomra], dec=random2[randomdec], ra_units='degrees', dec_units='degrees')

        dd = treecorr.NNCorrelation(**corrconfig)
        dr = treecorr.NNCorrelation(**corrconfig)
        rd = treecorr.NNCorrelation(**corrconfig)
        rr = treecorr.NNCorrelation(**corrconfig)

        dd.process(DataCat1, DataCat2)
        dr.process(DataCat1, RandCat2)
        rd.process(DataCat2, RandCat1)
        rr.process(RandCat1, RandCat2)
        xi, varxi = dd.calculateXi(rr, dr, rd=rd)


    r = np.exp(dd.logr) / 60.0

    return [ [xi], [r] ]




def JelenaResults(covfile, wfile):
    cov = np.loadtxt(covfile)
    warr = np.loadtxt(wfile)
    xi = warr[:,1]
    r = warr[:,0]
    icov = np.linalg.pinv(cov)
    #print np.dot(cov, icov)
    return r, xi, cov, icov


def GetFromDisk(dir, coords='coords.txt', xi='xi.txt', cov='cov.txt', rcond=1e-15, njack=1):
    r = np.loadtxt(os.path.join(dir, coords))
    xi = np.loadtxt(os.path.join(dir, xi))
    cov = np.loadtxt(os.path.join(dir, cov)) / njack
    icov = np.linalg.pinv(cov, rcond=rcond)
    #icov = np.linalg.inv(cov)
    return r, xi, cov, icov





def PowerLaw(r, amp, exp):
    return amp * np.power(r, exp)
    #return np.power(r/amp, exp)


def LnPowerLaw(logr, amp, exp):
    return exp * (logr - amp)

def Chi2Array1D(params, function, x,data,icov):
    fit = function(x, *params)
    diff = data - fit
    chiarr = diff * np.dot(icov,diff) 
    return chiarr

def FitPowerLaw(r, xi, icov, function, guess):
    out = leastsq(Chi2Array1D, guess, args=(function,r,xi,icov), full_output=1)
    return out

def PlotCorr(r, xi, cov, ax=None, plotkwargs={}, fitkwargs={}, fit=None, fitparams=None, text=None, textkwargs=[], flatten=False):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(9,6))
    err = np.sqrt( np.diag(cov) )

    if flatten:
        xi = xi * np.power(r, 0.7)
        err = err * np.power(r, 0.7)

    ax.errorbar(r, xi, yerr=err, **plotkwargs)
    if fit is not None:
        for i in range(len(fit)):
            morer = np.linspace(np.amin(r), np.amax(r), 100)
            pts = fit[i](r, *fitparams[i])

            if flatten:
                pts = pts * np.power(r, 0.7)

            ax.plot(r, pts, **fitkwargs[i])

    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [deg]')

    if flatten:
        ax.set_ylabel(r'$\theta^{0.7} w \left( \theta \right)$')
    else:
        ax.set_ylabel(r'$w \left( \theta \right)$')

    if text is not None:
        for t,i in zip(text,range(len(text))):
            ax.text(t[0][0], t[0][1], t[1], verticalalignment='top', horizontalalignment='left', **textkwargs[i] )


def Tlnlike(params, x, data, tmatrix, pos):
    if np.abs(params[1] > 2):
        return -np.inf
    if np.abs(params[0] > 2):
        return -np.inf

    for p in pos:
        if params[p] <= 0:
            return -np.inf
    fit = PowerLaw(x, params[0], params[1])
    diff = data - fit
    Tdiff = np.dot(tmatrix, diff)
    lnprob = -0.5 * np.dot(diff.T, Tdiff)
    return lnprob


#def lnlike(params, function, x, data, icov, pos, doprint=False):
def lnlike(params, x, data, icov, pos, doprint):
    if np.abs(params[1] > 2):
        return -np.inf
    if np.abs(params[0] > 2):
        return -np.inf

    for p in pos:
        if params[p] <= 0:
            return -np.inf
    
    fit = PowerLaw(x, params[0], params[1])
    diff = data - fit
    lnprob = -0.5 * np.dot(np.transpose(diff), np.dot(icov, diff))
    #lnprob = -np.dot(diff.T, np.dot(icov, diff))

    if doprint:
        pass
   
    '''
    if lnprob < -100:
        lnprob = -np.inf
    '''
    
    return lnprob


#def MCMC(r, xi, icov, function, guess, nwalkers=1000, nburn=1000, nsample=500, pos=[]):
def MCMC(r, xi, icov, cov, guess, nwalkers=1000, nburn=1000, nsample=500, pos=[], alpha2=1e-10, tik=False):
    ndim = len(guess)
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(function, r, xi, icov, pos))

    if not tik:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(r, xi, icov, pos, False))

    else:
        tmatrix = Utils.TikhonovMatrix(cov, alpha2=alpha2)
        print icov[-1]
        print tmatrix[-1]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Tlnlike, args=(r, xi, tmatrix, pos))

    #initguess = np.random.beta(2, 5, size=(nwalkers,ndim))/1.0e4 + np.reshape(guess, (1,guess.shape[0]))
    #s = np.amin(guess) / 1.0e6
    ##s = np.abs( np.amin(guess) / 1.0e2 )
    #s = 1
    #s = 1e-4
    #initguess = np.random.beta(2, 5, size=(nwalkers, ndim)) * s + np.reshape(guess, (1,guess.shape[0]))

    initguess = np.reshape(np.random.randn(nwalkers), (nwalkers,1)) * (1.0e-5*np.abs(guess)) + np.reshape(guess, (1,guess.shape[0]))
    pos, prob, state = sampler.run_mcmc(initguess, nburn)
    sampler.reset()
    pos, prob, stat = sampler.run_mcmc(pos, nsample)

    f = sampler.flatchain
    amp = np.average(f[:,0])
    err_amp = np.std(f[:,0])
    exp = np.average(f[:,1])
    err_exp = np.std(f[:,1])

    fig, ax = plt.subplots(1,1, figsize=(9,6))
    expp = f[:, 1]
    ax.plot(np.arange(expp.shape[0]),expp)

    fig, ax = plt.subplots(1,1, figsize=(9,6))
    ampp = f[:, 0]
    print np.average(ampp), np.std(ampp)
    ax.plot(np.arange(ampp.shape[0]),ampp)

    fig, ax = plt.subplots(1,1, figsize=(9,6))
    p = sampler.flatlnprobability
    ax.plot(np.arange(p.shape[0]),p)
    max = np.argmax(p)
    m = f[max,:]

    cov = np.sum((ampp - amp) * (expp-exp)) / len(f)
    #print exp, err_exp, amp, err_amp, p[max]*2/float(len(r)-2)
    #print np.average(sampler.acceptance_fraction)

    obj = np.zeros(nwalkers*nsample, dtype=[('lnprob',np.float32), ('amp',np.float32), ('exp',np.float32)])
    obj['lnprob'] = p
    obj['amp'] = ampp
    obj['exp'] = expp
    obj = np.sort(obj, order='lnprob')[::-1]
    #print obj['lnprob']
    #print obj['exp']
    #print obj['amp']

    c = int( len(obj)*0.68 )
    cc = int( len(obj)*0.95 )
    h = obj[0:c]
    h2 = obj[c:cc]

    hnull, abins, ebins = np.histogram2d(obj['amp'], obj['exp'], bins=(100,100))
    extent = [abins[0],abins[-1],ebins[0],ebins[-1]]
    hnull[:,:] = 0

    h, abins, ebins = np.histogram2d(h['amp'], h['exp'], bins=[abins,ebins])
    cut = (h > 0)
    hnull[cut] = 1

    h, abins, ebins = np.histogram2d(h2['amp'], h2['exp'], bins=[abins,ebins])
    cut = (h > 0)
    hnull[cut] = 2

    fig, ax = plt.subplots(1,1, figsize=(9,6))
    cax = ax.imshow(hnull.T, origin='lower', interpolation='nearest', extent=extent, aspect='auto')



    return amp, err_amp, exp, err_exp
    #return m[0], err_amp, m[1], err_exp

def SimulatedPowerLaw(amp=1e-3, exp=-0.7, noise=np.linspace(0.05,0.50,30), r=None, cov=None):

    if r is None:
        r = np.linspace(np.log(0.01), np.log(5), 30)
        r = np.exp(r)

    s = amp * np.power(r, exp)
    if cov is None:
        noiselevel = noise * s
        cov = np.diag(noiselevel*noiselevel) + np.reshape(-0.02*noiselevel*noiselevel, (1,noiselevel.shape[0]))

    nn = np.random.multivariate_normal( np.zeros(len(r)), cov)
    #n = np.random.randn(len(s)) * noiselevel
    noised = s + nn

    #cov = np.diag( np.power(noiselevel, 2) )
    #icov = np.linalg.pinv(cov)
    #return r, noised, cov, icov
    return r, noised, cov



def MCMCFit(x, y, cov, fitl=1e-3, fitu=10, alpha2=1e-15, rcond=1e-15, tik=False):

    guess = np.array([2.5e-3, -0.4])

    range = (x >= fitl) & (x <= fitu)
    ind = np.arange(len(x))[range]
    min = np.amin(ind)
    max = np.amax(ind) + 1
    x = x[range]
    y = y[range]
    cov = cov[ min:max, min:max ]
    #cov = np.diag(np.diag(cov))

    icov = np.linalg.pinv(cov, rcond=rcond)
    #icov = np.linalg.inv(cov)

    '''
    vmax = -5
    vmin = -10
     
    fig, axarr = plt.subplots(1,1, figsize=(12,4))
    ax = axarr[0]
    cax = ax.imshow(np.log10(cov_balrog), interpolation='nearest', origin='lower', extent=[np.log10(r_balrog[0]),np.log10(r_balrog[-1]), np.log10(r_balrog[0]),np.log10(r_balrog[-1])], vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=axarr[0]) 
    ax.set_xlabel(r'$\log_{10} \theta$')
    ax.set_title('Condition number = %.3e'%(np.linalg.cond(cov_balrog)))

    plt.tight_layout()
    #plt.show()
    '''

    #bamp, bamp_err, bexp, bexp_err = MCMC(r_balrog, xi_balrog, icov_balrog, PowerLaw, guess, pos=[0])
    #amp, amp_err, exp, exp_err = MCMC(x, y, icov, guess, pos=[0])
    amp, amp_err, exp, exp_err = MCMC(x, y, icov, cov, guess, pos=[0], alpha2=alpha2, tik=tik)
    params = [amp,exp]
    eparams = [amp_err,exp_err]

    if not tik:
        #chi2 = -2 * lnlike(params, PowerLaw, x, y, icov, [], doprint=True) / (len(x) - len(params))
        chi2 = -2 * lnlike(params, x, y, icov, [], True) / (len(x) - len(params))
    else:
        tmatrix = Utils.TikhonovMatrix(cov, alpha2=alpha2)
        chi2 = -2 * Tlnlike(params, x, y, tmatrix, []) / (len(x) - len(params))

    fit = PowerLaw(x, *params)
    res = y - fit


    return params, eparams, chi2


    """
    #fig, ax = plt.subplots(2,2, figsize=(16,9))
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(2,2, height_ratios=[3,1])
    ax = np.array([ [plt.subplot(gs[0]),plt.subplot(gs[1])], [plt.subplot(gs[2]),plt.subplot(gs[3])] ])

    #bloc = (0.5, 0.05)
    #uloc = (0.5, 0.03)
    bloc = (0.07, 0.1)
    uloc = (0.07, 0.05)
    cloc = 0.02

    eq1 = r'$w \left( \theta \right) = A \theta^{\alpha}$' + '\n' + r'$A = %.2e \pm %.2e$' %(bamp,bamp_err) + '\n' r'$\alpha = %.2e \pm %.2e$' %(bexp,bexp_err)
    eq2 = r'$w \left( \theta \right) = A \theta^{\alpha}$' + '\n' + r'$A = %.2e \pm %.2e$' %(uamp,uamp_err) + '\n' r'$\alpha = %.2e \pm %.2e$' %(uexp,uexp_err)
    PlotCorr(r_balrog, xi_balrog, cov_balrog, ax=ax[0,0], plotkwargs={'color':'blue', 'label':'Benchmark: Balrog Random', 'fmt':'.'}, fit=[PowerLaw,PowerLaw], fitparams=[bparams,uparams], fitkwargs=[{'color':'blue', 'ls':'dashed'},{'color':'red', 'ls':'dashed'}], text=[ [bloc, eq1], [uloc, eq2]] , textkwargs=[{'color':'blue'}, {'color':'red'}], flatten=False)
    ax[0,0].set_title('Balrog Random', color='blue')
    ax[0,0].text(cloc, bloc[1], r'$\chi^2 / \mathrm{DOF} = %.2f$' %(b_chi2), verticalalignment='top', horizontalalignment='left', color='blue')

    PlotCorr(r_uniform, xi_uniform, cov_uniform, ax=ax[0,1], plotkwargs={'color':'red', 'label':'Benchmark: Uniform Random', 'fmt':'.'}, fit=[PowerLaw,PowerLaw], fitparams=[bparams,uparams], fitkwargs=[{'color':'blue', 'ls':'dashed'},{'color':'red', 'ls':'dashed'}], text=[ [bloc, eq1], [uloc, eq2]] , textkwargs=[{'color':'blue'}, {'color':'red'}], flatten=False)
    ax[0,1].set_title('Uniform Random', color='red')
    ax[0,1].text(cloc, uloc[1], r'$\chi^2 / \mathrm{DOF} = %.2f$' %(u_chi2), verticalalignment='top', horizontalalignment='left', color='red')


    ax[0,0].text(1, bloc[1], r'$A = %.2e$' %(5.0e-3) + '\n' + r'$\alpha = %.2e$' %(-0.59), verticalalignment='top', horizontalalignment='left', color='blue')
    ax[0,1].text(1, uloc[1], r'$A = %.2e$' %(3.0e-3) + '\n' + r'$\alpha = %.2e$' %(-0.65), verticalalignment='top', horizontalalignment='left', color='red')


    ax[1,0].errorbar(r_balrog, bres, yerr=np.sqrt(np.diag(cov_balrog)), fmt='o', color='blue')
    ax[1,0].axhline(y=0, color='blue')
    ax[1,1].errorbar(r_uniform, ures, yerr=np.sqrt(np.diag(cov_uniform)), fmt='o', color='red')
    ax[1,1].axhline(y=0, color='red')

    ax[0,0].axhline(y=0, color='blue')
    ax[0,1].axhline(y=0, color='red')


    ax[0,0].set_ylim( [1e-3, 3e-1] )
    ax[0,1].set_ylim( [1e-3, 3e-1] )
    #ax[0,0].set_ylim( [-0.004, 0.004] )
    #ax[0,1].set_ylim( [-0.004, 0.004] )

    ax[1,0].set_ylim( [-0.005, 0.005] )
    ax[1,1].set_ylim( [-0.005, 0.005] )

    xlow = 0.001
    xhigh = 1
    ax[0,0].set_xlim( [xlow, xhigh] )
    ax[0,1].set_xlim( [xlow, xhigh] )
    ax[1,0].set_xlim( [xlow, xhigh] )
    ax[1,1].set_xlim( [xlow, xhigh] )
    
    
    ax[1,0].set_xscale('log')
    ax[1,1].set_xscale('log')

    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')

    plt.tight_layout()
    plt.show(block=True)
    """


def UniformRandom(size=5e8, band='i', rakey='ra', deckey='dec'):
    ramin = 60.0,  
    ramax = 95.0
    decmin = -62.0
    decmax = -42.0

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

    uniform = np.zeros(len(ra), dtype=[('%s_%s'%(rakey, band),np.float64), ('%s_%s'%(deckey, band),np.float64)])
    uniform['%s_%s'%(rakey, band)] = ra
    uniform['%s_%s'%(deckey, band)] = dec
    
    #return ra, dec
    return uniform



def sgcut(arr, val, band='i'):
    cut = (arr['modest_%s'%(band)]==val)
    return arr[cut]

def mcut(arr, l, u, col='mag_auto', band='i'):
    if band is not None:
        col = '%s_%s' %(col,band)
    cut = (arr[col] > l) & (arr[col] < u)
    return arr[cut]



def Correlate(band='i', outdir='CorrFiles', outlabel='test', jfile='24-jacks.txt', njack=24, mag=True, generatejack=False,  corrconfig=None, lower=None, upper=None, modest=0, declow=None, dechigh=None, vers=None, invertvers=False, rtype='n-1', killnosim=True):
    '''
    print 'getting masks'
    speedup = Utils.GetUsualMasks()
    datadir = os.path.join(os.environ['GLOBALDIR'],'sva1-umatch')

    print 'reading data'
    sim, des, nosim = Utils.GetData2(band='i', dir=datadir, killnosim=False, notruth=True, needonly=False)

    print 'generating uniform'
    uniform = UniformRandom(size=10e7, band=band, rakey='ra', deckey='dec')


    #print len(uniform), len(sim), len(des)
    print 'making cuts'
    sim = Utils.ApplyThings(sim, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=True, lower=lower, upper=upper, colorcut=True, badflag=True, jbadflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, declow=declow, dechigh=dechigh, nozero=False, noduplicate=False, vers=vers, invertvers=invertvers, **speedup)
    des = Utils.ApplyThings(des, band, slr=True, slrinvert=False, slrwrite=None, modestcut=modest, mag=True, lower=lower, upper=upper, colorcut=True, badflag=True, jbadflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', posonly=False, declow=declow, dechigh=dechigh, vers=vers, invertvers=invertvers, **speedup)
    uniform = Utils.ApplyThings(uniform, band, slr=False, slrinvert=False, slrwrite=None, modestcut=None, mag=False, colorcut=False, badflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', ra='ra', dec='dec', declow=declow, dechigh=dechigh, **speedup)
    '''

    '''
    speedup = Utils.GetUsualMasks()
    uniform = UniformRandom(size=10e6, band=band, rakey='ra', deckey='dec')
    uniform = Utils.ApplyThings(uniform, band, slr=False, slrinvert=False, slrwrite=None, modestcut=None, mag=False, colorcut=False, badflag=False, elimask=True, benchmask=True, invertbench=False, posband='i', ra='ra', dec='dec', declow=declow, dechigh=dechigh, **speedup)
    print len(uniform)
    '''


    sim = esutil.io.read('sim4huff.fits')
    des = esutil.io.read('des4huff.fits')
    print len(sim), len(des)

    dstar = sgcut(des, 1)
    sstar = sgcut(sim, 1)
    dgal = sgcut(des, 0)
    sgal = sgcut(sim, 0)
    print len(dstar), len(dgal)

    sbad = (sstar['objtype_i']==1)
    print np.sum(sbad) / float(len(sstar))

    print len(dstar)/float(len(des)), len(sstar)/float(len(sim))
    print len(dgal)/float(len(des)), len(sgal)/float(len(sim))


    dmstar = mcut(dstar, 21, 23)
    smstar = mcut(sstar, 21, 23)
    print 'star S, D:', len(smstar), len(dmstar)
    
    dmgal = mcut(dgal, 23, 24)
    smgal = mcut(sgal, 23, 24)
    print 'gal S, D:', len(smgal), len(dmgal)

    '''
    fig, axarr = plt.subplots(1, 2, figsize=(16,8))
    npoints, xlim, ylim = Utils.PointMap(smgal, downfactor=3, band=band, ax=axarr[0], plotkwargs={'lw':0, 's':0.2})
    npoints, xlim, ylim = Utils.PointMap(dmgal, downsize=npoints, band=band, ax=axarr[1], plotkwargs={'lw':0, 's':0.2})
    plt.show()
    '''


    gbad = (smgal['objtype_i']==3)
    print np.sum(gbad) / float(len(smgal))

    sbad = (smstar['objtype_i']==1)
    print np.sum(sbad) / float(len(smstar))

    '''
    fig, axarr = plt.subplots(1,2, figsize=(12,6))
    bins = np.arange(19, 22.2, 0.1)
    axarr[0].hist(dmstar['mag_auto_i'], bins=bins, color='blue')
    axarr[0].hist(smstar['mag_auto_i'], bins=bins, color='red')
    plt.show()
    sys.exit()
    '''

    ra = 'alphawin_j2000_%s'%(band)
    dec = 'deltawin_j2000_%s'%(band)
    rra = 'ra_%s'%(band)
    rdec = 'dec_%s'%(band)

    basedir = os.path.join(outdir, outlabel, band)
    if generatejack:
        jtype = 'generate'
    else:
        jtype = 'read'


    '''
    save = FileSetup(basedir, 'SxG', corrconfig)
    #hists, covs, extra, jextra = JK.JackknifeOnSphere( [dgal,sgal, dstar,sstar], [ra,ra, ra,ra], [dec,dec, dec,dec], JKCorr, jargs=[], jkwargs={'cross':True, 'datara':ra, 'datadec':dec, 'randomra':ra, 'randomdec':dec, 'corrconfig':corrconfig}, jtype=jtype, njack=njack, jfile=jfile, save=save)
    hists, covs, extra, jextra = JK.AltJackknifeOnSphere( [dmgal,smgal, dmstar,smstar], [ra,ra, ra,ra], [dec,dec, dec,dec], JKCorr, jargs=[], jkwargs={'cross':True, 'datara':ra, 'datadec':dec, 'randomra':ra, 'randomdec':dec, 'corrconfig':corrconfig}, jtype=jtype, njack=njack, jfile=jfile, save=save, itsave=True, rtype=rtype)
    '''

    save = FileSetup(basedir, 'SS', corrconfig)
    hists, covs, extra, jextra = JK.AltJackknifeOnSphere( [dmstar,smstar], [ra,ra], [dec,dec], JKCorr, jargs=[], jkwargs={'cross':False, 'datara':ra, 'datadec':dec, 'randomra':ra, 'randomdec':dec, 'corrconfig':corrconfig}, jtype=jtype, njack=njack, jfile=jfile, save=save, itsave=True, rtype=rtype)

    #save = FileSetup(basedir, 'SU', corrconfig)
    #hists, covs, extra, jextra = JK.JackknifeOnSphere( [dstar,uniform], [ra,rra], [dec,rdec], JKCorr, jargs=[], jkwargs={'cross':False, 'datara':ra, 'datadec':dec, 'randomra':rra, 'randomdec':rdec, 'corrconfig':corrconfig}, jtype=jtype, njack=njack, jfile=jfile, save=save)



def FileSetup(basedir, name, corrconfig):
    save = os.path.join(basedir, name)
    if not os.path.exists(save):
        os.makedirs(save)
    jsonfile = os.path.join(save, 'config.json')
    with open(jsonfile, 'w') as outfile:
        json.dump(corrconfig, outfile)
    return save


def DepthSelect(arr, ra, dec, depth, lower=22.5, upper=23.5):
    hps = Utils.RaDec2Healpix(arr[ra], arr[dec], depth['nside'], nest=depth['nest'])
    vals = depth['map'][hps]
    cut = (vals > lower) & (vals < upper)
    return arr[cut]


def GetDepthMap():
    depthfile = 'sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits.gz'
    depthnest = True
    depthmap = hp.read_map(depthfile, nest=depthnest)
    depthnside = hp.npix2nside(depthmap.size)
    return {'map':depthmap, 'nest':depthnest, 'nside':depthnside}


def MakeDepthHist(des, ra, dec, depth):
    hps = Utils.RaDec2Healpix(des[ra], des[dec], depth['nside'], nest=depth['nest'])
    uhps = np.unique(hps)
    vals = depth['map'][uhps]
    cut = (vals > 0)
    vals = vals[cut]

    bins = np.arange(21, 25.0, 0.5)
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.hist(vals, bins=bins)
    ax.set_xlabel('mag')
    plt.show()



def PlotFromDisk(outlabel, band='i', outdir='CorrFiles', fig=None, axarr=None, label=[r'\textsc{Balrog}', r'Uniform'], color=['red','blue'], pfig=None, paxarr=None):
    basepath = os.path.join(outdir, outlabel, band)

    DB = Utils.GetVecInfo(os.path.join(basepath, 'DB'))
    DU = Utils.GetVecInfo(os.path.join(basepath, 'DU'))

    dbx = Utils.OffsetX(DB[2], log=True, offset=0.02)
    dby, dbe = Utils.TimesPowerLaw(DB[2], DB[0], np.sqrt(np.diag(DB[1])), 0.55)
    duy, due = Utils.TimesPowerLaw(DU[2], DU[0], np.sqrt(np.diag(DU[1])), 0.55)
 
   
    if fig is None:
        fig = plt.figure(figsize=(8, 6), tight_layout=True)
        gs = gridspec.GridSpec(1,1)
        axarr = []
        axarr.append( fig.add_subplot(gs[0]) )

    axarr[0].errorbar( DB[2], DB[0], yerr=np.sqrt(np.diag(DB[1])), color=color[0], label=label[0], fmt='o')
    axarr[0].errorbar( DU[2], DU[0], yerr=np.sqrt(np.diag(DU[1])), color=color[1], label=label[1], fmt='o')

    '''
    alpha2 = 1e-15
    #tmatrix = Utils.TikhonovMatrix(DB[1], alpha2=alpha2)
    #c = np.linalg.pinv(tmatrix)
    #DB[2], DB[0], DB[1] = SimulatedPowerLaw(r=DB[2], cov=c, amp=4e-3, exp=-0.59)
    #DB[2], DB[0], DB[1] = SimulatedPowerLaw(r=DB[2], cov=DB[1], amp=4e-3, exp=-0.59)
    #DB[1] = np.diag(np.diag(DB[1]))
    
    b_params, berr_params, b_chi2 = MCMCFit(DB[2], DB[0], DB[1], fitl=2e-2, fitu=1, alpha2=alpha2, tik=False, rcond=1e-15)
    #b_params, berr_params, b_chi2 = MCMCFit(DB[2], DB[0], DB[1], fitl=2e-3, fitu=1, alpha2=alpha2, tik=False, rcond=1e-15)
    b_fit = PowerLaw(DB[2], *b_params)
    b_res = DB[0] - b_fit
    axarr[0].plot( DB[2], b_fit, color='red', ls='--')

    bloc = [3e-3, 2e-3]
    bstr = r'$w(\theta) = A \, \theta^{\alpha}$' + '\n' + r'$\alpha = %.2e \pm %.2e$' %(b_params[1], berr_params[1]) + '\n' + r'$A = %.2e \pm %.2e$' %(b_params[0],berr_params[0])
    axarr[0].text(bloc[0], bloc[1], bstr, color='red', verticalalignment='top', horizontalalignment='left')

    print b_params
    print b_chi2
    '''

    axarr[0].set_xscale('log')
    axarr[0].legend(loc='best')
    axarr[0].set_xlabel(r'$\theta$ [deg]')
    axarr[0].set_ylabel(r'$w(\theta)$')
    axarr[0].set_yscale('log')

    axarr[0].set_ylim([0.001, 2])
    axarr[0].set_xlim([0.0003, 1]) 

    if pfig is None:
        pfig = plt.figure(figsize=(9, 6), tight_layout=True)
        gs = gridspec.GridSpec(1,1)
        paxarr = []
        paxarr.append( pfig.add_subplot(gs[0]) )

    paxarr[0].errorbar( dbx, dby, yerr=dbe, color=color[0], label=label[0], fmt='o')
    paxarr[0].errorbar( DU[2], duy, yerr=due, color=color[1], label=label[1], fmt='o')
    paxarr[0].set_xscale('log')
    paxarr[0].legend(loc='best')
    paxarr[0].set_xlabel(r'$\theta$ [deg]')
    paxarr[0].set_ylabel(r'$\theta^{0.7} \, w(\theta)$')

    return fig, axarr, pfig, paxarr




if __name__=='__main__': 

    style = Utils.ReadStyle('custom-sytle.mpl')
    style = Utils.SetStyle(style)

    band = 'i'
    corrconfig = {'sep_units': 'arcmin',
                  'min_sep': 0.06,
                  'max_sep': 600.0,
                  'nbins': 45,
                  'bin_slop': 0.25}


    #label = 'SxG-test-21-22'
    #label = 'SS-test-U'

    #label = 'SxG-test-23-24'
    #Correlate(band=band, corrconfig=corrconfig, outlabel=label, jfile=os.path.join('JK-regions', '24JK-%s'%(label)), generatejack=False)
    label = 'SxG-23-24-DEC-3'
    #Correlate(band=band, corrconfig=corrconfig, outlabel=label, jfile=os.path.join('JK-regions', '24JK-BM-23-24-DEC'), generatejack=False)
    #sys.exit()

    fig, ax = plt.subplots(1,1, figsize=(10,8))

    '''
    label = 'SxG-test-21-22'
    sg_theta, sg_w, sg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='SxG', plotkwargs={'c':'cyan', 'label':r'SxG, G21', 'fmt':'o'}, ax=ax, get=True, contam=0, noplot=True)
    '''

    #label = 'SxG-test'
    #label = 'SxG-test-23-24'
    #sg_theta, sg_w, sg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='SxG', plotkwargs={'c':'green', 'label':r'SxG, G23', 'fmt':'o'}, ax=ax, get=True, contam=0, noplot=True)

    label = 'SxG-23-24-DEC-3'
    sg_theta, sg_w, sg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='SxG', plotkwargs={'c':'blue', 'label':r'SxG', 'fmt':'o'}, ax=ax, get=True, contam=0, noplot=True)

    #sg_w[:] = 1.5e-3

    #label = 'SS-test'
    ss_theta, ss_w, ss_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='SS', plotkwargs={'c':'cyan', 'label':r'SS Direct', 'fmt':'o'}, ax=ax, get=True, contam=0)

    '''
    label = 'SS-test-U'
    ss_theta, ss_w, ss_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='SU', plotkwargs={'c':'blue', 'label':r'SU $19 < i < 21$', 'fmt':'o'}, ax=ax, get=True, contam=0)
    '''
   
    label = 'BM-23-24-DEC'
    gg_theta, gg_w, gg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='DB', plotkwargs={'c':'red', 'label':r'DES \textsc{Balrog} (Raw)', 'fmt':'o'}, ax=ax, get=True, contam=0, noplot=True)

    #ax.scatter(gg_theta, gg, color='blue')
    #gg_theta, gg_w, gg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='DB', plotkwargs={'c':'orange', 'label':r'DES \textsc{Balrog} (Corrected)', 'fmt':'o'}, ax=ax, get=True, contam=0)
    #gg_theta, gg_w, gg_cov, gg = MoreUtils.RecreateContam(label, njack=24, gc=0.04, sc=0.04, band='i', kind='DB')
    #ax.errorbar(gg_theta, gg_w, np.sqrt(np.diag(gg_cov)), fmt='o', color='orange', label='Corrected')
    #ax.scatter(gg_theta, gg, color='blue')
    #gg_theta, gg_w, gg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='DB', plotkwargs={'c':'orange', 'label':r'DES \textsc{Balrog} (Corrected)', 'fmt':'o'}, ax=ax, get=True, contam=0)

    gc = 0.07
    sc = 0.03
    
    cor = MoreUtils.ContaminationCorrection(gg_w, sg_w, gc=gc, sc=sc)
    ss = MoreUtils.StellarAuto(gg_w, sg_w, gc=gc, sc=sc)
    cor2 = MoreUtils.AshleyCorrection(gg_w, ss_w, gc=gc)

    #label = 'BM-23-24-DEC-corrected-4'
    label = 'BM-23-24-full'
    #gg_theta, gg_w, gg_cov = MoreUtils.RecreateContam(label, njack=24, gc=gc, sc=sc, band='i', kind='DB')
    gg_theta, gg_w, gg_cov = PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='DB', plotkwargs={'c':'orange', 'label':r'SG Cor', 'fmt':'o'}, ax=ax, get=True, contam=0, noplot=False)

    f_theta, f_ss, f_sscov = MoreUtils.GetThing(label, 4, other=0, njack=24, outdir='CorrFiles', band='i', kind='DB')
    f_theta, f_iss, f_isscov = MoreUtils.GetThing(label, 5, other=0, njack=24, outdir='CorrFiles', band='i', kind='DB')
    f_theta, f_gg, f_ggcov = MoreUtils.GetThing(label, 1, other=0, njack=24, outdir='CorrFiles', band='i', kind='DB')
    cor3 = MoreUtils.AshleyCorrection(f_gg, f_ss, gc=gc)

    ax.errorbar(f_theta, f_ss, yerr=np.sqrt(np.diag(f_sscov)), fmt='o', color='maroon', label='Full SS')
    ax.errorbar(f_theta, f_iss, yerr=np.sqrt(np.diag(f_isscov)), fmt='o', color='pink', label='Full ISS')
    ax.errorbar(f_theta, f_gg, yerr=np.sqrt(np.diag(f_ggcov)), fmt='o', color='red', label='Full Raw')
    ax.scatter(f_theta, cor3, color='blue', label='Full ACor')

    gg_theta, gg_w, gg_cov = MoreUtils.RecreateContam(label, njack=24, gc=0.2, sc=0.04, band='i', kind='DB')

    '''
    if gc != 0:
        ss_w = (sg_w - sc* gg_w)
        #ax.scatter(gg_theta, ss_w, color='blue', label='My guess at SS')
        cor = np.power(1 + gc, 2) * ( gg_w - gc*ss_w - np.power(gc,4) / np.power(1+gc, 2) )

        #cor = np.power(1 + gc, 2) * ( gg_w - gc*gc* ss_w - np.power(gc,4) / np.power(1+gc, 2) )
    else:
        cor = gg_w
    '''

    #ax.errorbar(gg_theta, cor, np.sqrt(np.diag(gg_cov)), color='orange', label='Stellar Contamination Corrected', fmt='o')
    #ax.errorbar(gg_theta, gg_w, np.sqrt(np.diag(gg_cov)), color='orange', label='Stellar Contamination Corrected', fmt='o')
    #ax.scatter(gg_theta, cor2, color='blue', label='Corrected')
    ax.scatter(gg_theta, gg_w, color='dodgerblue', label='Corrected')
    ax.scatter(gg_theta, ss, color='purple', label='SS from SxG')

    e = np.loadtxt('wtheta_EH_z2_z1BAOnlirealpz-suchyta.txt')
    b = 1.18
    ax.plot(e[:,0]/60.0, b*b*e[:,1], color='gray', ls='--')
    #b = 1.0
    #ax.plot(e[:,0]/60.0, b*b*e[:,1], color='gray', ls='-.')

    '''
    e = np.loadtxt('wtheta_EH_z2_z1BAOlinrealpz-suchyta.txt')
    b = 1.0
    ax.plot(e[:,0]/60.0, b*b*e[:,1], color='gray', ls='--')
    '''

    #e = np.loadtxt('w-enrique.txt')
    #ax.errorbar(e[:,0]/60.0, e[:,1], yerr=e[:,2], color='green', fmt='o', label="Enrique's COSMOS Measurement")

    #label = 'COSMOS-Umorph-23-24-v1'
    #PlottingStuff.PlotOne(label, outdir='CorrFiles', band='i', kind='DU', plotkwargs={'c':'black', 'label':r"Eric's COSMOS Measurement", 'fmt':'o'}, ax=ax)


    ax.legend(loc='lower left')
    #ax.legend(loc='best')
    plt.show()
