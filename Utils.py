import os
import sys
import esutil
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import healpy as hp
import slr_zeropoint_shiftmap
from scipy.optimize import fsolve as fsolve
import numpy.lib.recfunctions as recfunctions
import kmeans_radec

def ReadStyle(file):
    d = {}
    with open(file) as f:
        for line in f:
            l = line.strip()
            if (l=='') or (l[0]== '#') :
                continue
            
            key, val = l.split(':')
            k = key.strip()
            v = val.strip()

            if v[0] in ["'", '"']:
                d[k] = v
            else:
                try:
                    d[k] = float(v)
                except:
                    d[k] = v
    return d

def SetStyle(style):
    for k,v in style.iteritems():
        matplotlib.rcParams[k]=v

def TikhonovMatrix(A, alpha2=1e-10):
    gamma = np.identity(len(A))
    i = np.dot(A.T, A) + alpha2*gamma
    inv = np.linalg.inv(i)
    #inv = np.linalg.pinv(i)
    v = np.dot(inv, A.T)
    return v


def Tikhonov(A, x, alpha2=1e-10):
    gamma = np.identity(len(x))
    i = np.dot(A.T, A) + alpha2*gamma
    inv = np.linalg.inv(i)
    #inv = np.linalg.pinv(i)
    v = np.dot(inv, np.dot(A.T, x))
    return v

def TChi2(diff, cov, alpha2=1e-10):
    idiff = Tikhonov(cov, diff, alpha2=alpha2)
    return np.dot(diff.T, idiff) / float(len(diff))


def TimesPowerLaw(x, y, err, exp):
    ynew = y * np.power(x, exp)
    errnew = err * np.power(x, exp)
    return ynew, errnew


def ChiFromZero(diff, cov, rcond=1e-15):
    icov = np.linalg.pinv(cov, rcond=rcond)
    #icov = np.linalg.inv(cov)
    chi2 = np.dot(diff.T, np.dot(icov,diff)) / float( len(diff) )

    #print diff
    #print np.dot(icov,diff)
    #print diff * np.dot(icov,diff)

    return chi2


def Chi2Disagreement(a1, c1, a2, c2, rcond=1e-15):

    diff = a1 - a2
    #cov = c1 + c2
    cov = c2 * np.sqrt(2)
    icov = np.linalg.pinv(cov, rcond=rcond)
    #icov = np.linalg.inv(cov)
    chi2 = np.dot(diff.T, np.dot(icov,diff)) / float( len(a1) )
    print len(a1)
    print diff
    print np.dot(icov,diff)
    
    eval, evec = np.linalg.eig(cov)
    print eval
    print np.linalg.cond(cov)
   
    '''
    diff = a1 - a2
    cov = c1 + c2
    root, debug = fsolve(csolve, 1,0, args=(cov,diff), full_output=1)
    print root, debug
    return root / float(len(a1))
    '''

    return chi2


def csovle(chi2, cov, d):
    dinv = 1.0 / (d * len(d))
    return chi2 * np.dot(dinv, np.dot(cov, dinv)) - 1


def ArcsinchImage(jcov, ax, fig, bins, plotscale=1e-4, nticks=9, title=None, axislabel=None):
    #nticks = 9
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

    cax = ax.imshow(jplot, origin='lower', interpolation='nearest', extent=[bins[0], bins[-1], bins[0], bins[-1]], vmin=-scale, vmax=scale, cmap=plt.get_cmap('bwr'), aspect='equal')
    cbar = fig.colorbar(cax, ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels(tlabels)

    if title is not None:
        ax.set_title(title)
    if axislabel is not None:
        ax.set_xlabel(axislabel)
        ax.set_ylabel(axislabel)


def GetData2(dir=os.path.join(os.environ['GLOBALDIR'],'v2+3_matched'), band='i', bands='griz', killnosim=False, notruth=False, needonly=False, noversion=False):
    need = ['mag_auto_g', 'mag_auto_r', 'mag_auto_i', 'mag_auto_z', 'modest_i', 'badflag', 'alphawin_j2000_i', 'deltawin_j2000_i']
    band_need = ['alphawin_j2000', 'deltawin_j2000']

    '''
    if not noversion:
        band_need.append('version')
    '''

    for b in band_need:
        bstr = '%s_%s'%(b,band)
        if bstr not in need:
            need.append(bstr)

    if not noversion:
        need.append('version')

    need.append('balrog_index')

    if not notruth:
        truth = esutil.io.read(os.path.join(dir, 'truth_%s-%s.fits'%(band,bands)))

    if needonly:
        matched = esutil.io.read(os.path.join(dir, 'matched_%s-%s.fits'%(band,bands)), columns=need)
        des = esutil.io.read(os.path.join(dir, 'des_%s-%s.fits'%(band,bands)), columns=need[:-1])
    else:
        matched = esutil.io.read(os.path.join(dir, 'matched_%s-%s.fits'%(band,bands)))
        des = esutil.io.read(os.path.join(dir, 'des_%s-%s.fits'%(band,bands)))

    nosim = esutil.io.read(os.path.join(dir, 'nosim_%s-%s.fits'%(band,bands)))

    a = len(matched)
    if killnosim:

        if noversion:
            kill = np.in1d(matched['balrog_index'], nosim['balrog_index'])
            matched = matched[-kill]

        else:
            #vstr = 'version_%s'%(band)
            vstr = 'version'
            bstr = 'balrog_index'
            versions = np.unique(nosim[vstr])
            for v in versions:
                mv_cut = (matched[vstr]==v)
                nv_cut = (nosim[vstr]==v)
                bad = nosim[nv_cut][bstr]
                mb_cut = np.in1d(matched[bstr],bad)
                both = (mv_cut & mb_cut)
                matched = matched[-both]

    if not notruth:
        return truth, matched, des, nosim
    else:
        return matched, des, nosim


def GetUsualMasks():
    elifile = 'sva1_gold_1.0.4_goodregions_04_equ_ring_4096.fits.gz'
    elinest = False
    elimap = hp.read_map(elifile, nest=elinest)
    benchfile = 'mask_healpix_nest_ns4096_sva1_gold_1.0.2-4_magautoi.ge.22p5_goodregions_04_fracdet.ge.0.8.dat'
    benchnest = True
    benchnside = 4096
    benchhps = np.int64( np.loadtxt(benchfile) )
    speedup = {'elimap':elimap, 'elinest':elinest, 'benchhps':benchhps, 'benchnest':benchnest, 'benchnside':benchnside}
    return speedup


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


def RaDec2Healpix(ra, dec, nside, nest=False):
    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    hpInd = hp.ang2pix(nside, theta, phi, nest=nest)
    return hpInd

def BenchMarkArea(cat, ra, dec, invertbench=False, file='mask_healpix_nest_ns4096_sva1_gold_1.0.2-4_magautoi.ge.22p5_goodregions_04_fracdet.ge.0.8.dat', nest=True, nside=4096, benchhps=None):
    if benchhps is None:
        hps = np.int64( np.loadtxt(file) )
    else:
        hps = benchhps
    pix = RaDec2Healpix(cat[ra], cat[dec], nside, nest=nest)
    cut = np.in1d(pix, hps)
    if invertbench:
        deccut = (cat[dec] > -60)
        return cat[(-cut)&(deccut)]
    else: 
        return cat[cut]


def GalaxyCut(cat, band, key='modtype'):
    cut = (cat['modtype_%s'%(band)]==0)
    return cat[cut]


def MagCut(cat, band, lower=18, upper=24):
    cut = (cat['mag_auto_%s'%(band)] > lower) & (cat['mag_auto_%s'%(band)] < upper)
    return cat[cut]

def OKRegions(cat, ra, dec, regionfile='sva1_gold_1.0.4_goodregions_04_equ_ring_4096.fits.gz', nest=False, elimap=None):
    if elimap is None:
        map = hp.read_map(regionfile, nest=nest)
    else:
        map = elimap
    nside = hp.npix2nside(map.size)
    pix = RaDec2Healpix(cat[ra], cat[dec], nside, nest=nest)
    use = (map[pix]==1)
    return cat[use]

def ColorCut(cat):
    gr = cat['mag_auto_g'] - cat['mag_auto_r']
    ri = cat['mag_auto_r'] - cat['mag_auto_i']
    iz = cat['mag_auto_i'] - cat['mag_auto_z']
    cut1 = (-1 < gr) & (gr < 3)
    cut2 = (-1 < ri) & (ri < 2)
    cut3 = (-1 < iz) & (iz < 2)
    cut = (cut1 & cut2 & cut3)
    return cat[cut]

def ModestCut(cat, band='i', key='modest', val=0):
    cut = (cat['%s_%s'%(key,band)]==val)
    return cat[cut]

def SLRCorrect(cat, ra, dec, bands=['g','r','i','z'], mag='mag_auto', slrfile='slr_zeropoint_shiftmap_v6_splice_cosmos_griz_EQUATORIAL_NSIDE_256_RING.fits', slrinvert=False, slrwrite=None):
    my_slr_map = slr_zeropoint_shiftmap.SLRZeropointShiftmap(slrfile, fill_periphery=True)

    for band in bands:
        m = '%s_%s'%(mag,band)
        m_shift, m_quality = my_slr_map.GetZeropoint(band, cat[ra], cat[dec], cat[m], interpolate=True)
        if slrinvert:
            cat[m] = cat[m] - m_shift
        else:
            cat[m] = cat[m] + m_shift

    if slrwrite is not None:
        esutil.io.write(slrwrite, cat, clobber=False)

    return cat


def BadFlag(cat, col='badflag'):
    cut = (cat[col] <= 1)
    return cat[cut]


def ApplyThings(cat, band, ra='alphawin_j2000', dec='deltawin_j2000', benchmask=False, invertbench=False, galaxy=False, mag=True, lower=18, upper=22.5, elimask=True, elimap=None, elinest=False, benchhps=None, benchnside=4096, benchnest=True, colorcut=False, modestcut=None, slr=False, slrfile='slr_zeropoint_shiftmap_v6_splice_cosmos_griz_EQUATORIAL_NSIDE_256_RING.fits', slrinvert=False, slrbands=['g','r','i','z'], slrwrite=None, badflag=False, posband=None, posonly=False, declow=None, dechigh=None, nozero=False, noduplicate=False, jbadflag=False, vers=None, invertvers=False):
    if posband is None:
        posband = band

    ra = '%s_%s' %(ra, posband)
    dec = '%s_%s' %(dec, posband)

    if slr:
        cat = SLRCorrect(cat, ra, dec, bands=slrbands, slrfile=slrfile, slrinvert=slrinvert, slrwrite=slrwrite)

    if galaxy:
        cat = Modestify(cat, byband=band)
        cat = GalaxyCut(cat, band)
    if modestcut is not None:
        cat = ModestCut(cat, band='i', key='modest', val=modestcut)

    if mag:
        cat = MagCut(cat, band, lower=lower, upper=upper)

    if colorcut:
        cat = ColorCut(cat)

    if badflag:
        cat = BadFlag(cat)
    if jbadflag:
        cat = BadFlag(cat, col='jbadflag')

    if nozero:
        print 'before nozero', len(cat)
        cut = (cat['flux_0_%s'%(band)]>0)
        cat = cat[cut]
        print 'after nozero', len(cat)

    if noduplicate:
        print 'before noduplicate', len(cat)

        versions = np.unique(cat['version'])
        for i in range(len(versions)):
            vcut = (cat['version']==versions[i])

            c = np.sort(cat[vcut], order='balrog_index')
            diff = np.diff(c['balrog_index'])
            dcut = (diff==0)
            bad = c['balrog_index'][1:][dcut]

            cut = (np.in1d(cat['balrog_index'],bad)) & (vcut)
            cat = cat[-cut]

            #rdiff = np.diff(cat['balrog_index'][::-1])
            #b2 = (rdiff==0)
            #b2 = np.insert(b2, 0, False)
            #b2 = b2[::-1]
            #bad = (b1 | b2)
            #cat = cat[-bad]
        
        print 'after noduplicate', len(cat)


    if elimask:
        cat = OKRegions(cat, ra, dec, elimap=elimap, nest=elinest)

    if benchmask:
        cat = BenchMarkArea(cat, ra, dec, invertbench=invertbench, benchhps=benchhps, nside=benchnside, nest=benchnest)

    if declow is not None:
        cut = (cat[dec] > declow)
        cat = cat[cut]
    if dechigh is not None:
        cut = (cat[dec] < dechigh)
        cat = cat[cut]

    if vers is not None:
        #cut = (sim['version_i']==vers)
        cut = (cat['version']==vers)

        if invertvers:
            cat = cat[-cut]
        else:
            cat = cat[cut]

    if posonly:
        c = np.zeros( len(cat), dtype=[(ra,np.float32), (dec,np.float32)] )
        c[ra] = cat[ra]
        c[dec] = cat[dec]
        return c

    return cat  


def AddColor(cat, key='mag_auto', bands=['i','z']):
    m1 = cat['%s_%s'%(key,bands[0])]
    m2 = cat['%s_%s'%(key,bands[1])]
    c = m1 - m2
    cat = recfunctions.append_fields(cat, '%s_%s%s'%(key,bands[0],bands[1]), c, usemask=False)
    return cat


def OffsetX(r, offset=0, log=False):
    if log:
        newr = np.log10(r)
    else:
        newr = copy.copy(r)
    
    newr = newr + offset
    if log:
        newr = np.power(10, newr)

    return newr


def NTicks(ax, nxticks=None, nyticks=None):
    if nyticks is not None:
        ax.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nyticks-1)) )
        ax.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nyticks-1)) )
    if nxticks is not None:
        ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nxticks-1)) )
        ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(nbins=(nxticks-1)) )


def GetVecInfo(path):
    vec = np.loadtxt(os.path.join(path, 'vec', '0.txt'))#[1:-1]
    cov = np.loadtxt(os.path.join(path, 'cov', '0.txt'))#[1:-1, 1:-1]
    vec_coord = np.loadtxt(os.path.join(path, 'other', '0.txt'))#[1:-1]
    return [vec, cov, vec_coord]


def PointMap(data, band='i', x='alphawin_j2000', y='deltawin_j2000', ax=None, plotkwargs={}, downfactor=None, downsize=None, title=None, xlim=None, ylim=None, xlabel=None, ylabel=None):
    if band is not None:
        x = '{0}_{1}'.format(x,band)
        y = '{0}_{1}'.format(y,band)
  
    if downfactor is not None:
        size = len(data) / downfactor
        keep = np.random.choice(len(data), size=size, replace=False)
    elif downsize is not None:
        keep = np.random.choice(len(data), size=downsize, replace=False)
    else:
        keep = np.ones(len(data), dtype=np.bool_)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(data[x][keep],data[y][keep], **plotkwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    return len(data[keep]), ax.get_xlim(), ax.get_ylim()


def JKRegionPlot(cat, jfile, band, ra='alphawin_j2000', dec='deltawin_j2000', seed=None, ax=None, cmap=None):
    cra = '%s_%s'%(ra,band) 
    cdec = '%s_%s'%(dec,band) 

    xlabel = r'%s'%(cra.upper().replace('_', '\_'))
    ylabel = r'%s'%(cdec.upper().replace('_', '\_'))

    centers = np.loadtxt(jfile)
    km = kmeans_radec.KMeans(centers)
    njack = len(centers)

    rd = np.zeros( (len(cat),2) )
    rd[:,0] = cat[cra]
    rd[:,1] = cat[cdec]
    index = km.find_nearest(rd)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,6))

    if cmap is None:
        '''
        cmap =  matplotlib.cm.jet(np.linspace(0, 1, njack))
        np.random.seed(seed)
        np.random.shuffle(cmap)
        '''
        num = np.zeros(njack)
        for i in range(njack):
            num[i] = np.sum(index==i)
        min = float( np.amin(num) )
        max = float( np.amax(num) )
        cmap = (num - min) / (max)
        

    for i in range(njack):
        cut = (index==i)
        c = cat[cut]
        #npoints, xlim, ylim = PointMap(c, band=band, x=ra, y=dec, ax=ax, downfactor=30, plotkwargs={'lw':0, 's':0.2, 'c':cmap[i]}, xlabel=xlabel, ylabel=ylabel)
        npoints, xlim, ylim = PointMap(c, band=band, x=ra, y=dec, ax=ax, plotkwargs={'lw':0, 's':0.2, 'c':cmap[i]}, xlabel=xlabel, ylabel=ylabel)




def JKSeries(basedir, cat, jfile, band, ra='alphawin_j2000', dec='deltawin_j2000', seed=None, type='DB', outdir='JKPlots'):
    fig = plt.figure(figsize=(22,6), tight_layout=True)

    nax = 3
    gs = gridspec.GridSpec(1,nax, width_ratios=[1,1,1])
    axarr = []
    for i in range(nax):
        axarr.append(fig.add_subplot(gs[i]))
   
    centers = np.loadtxt(jfile)
    njack = len(centers)
    cmap =  matplotlib.cm.jet(np.linspace(0, 1, njack))
    np.random.seed(seed)
    mix = np.random.choice(njack, size=njack, replace=False)
    cmap = cmap[mix]

    area = JKRegionPlot(cat, jfile, band, ra=ra, dec=dec, seed=seed, cmap=cmap, ax=axarr[0])

    dir = os.path.join(basedir, type)
    vdir = os.path.join(dir, 'vec', 'it', '3')
    cdir = os.path.join(dir, 'other', 'it', '0')
    issdir = os.path.join(dir, 'vec', 'it', '4')

    axarr[1].set_xscale('log')
    axarr[1].set_yscale('log')
    #axarr[1].set_ylim( [1e-4, 1e-1] )
    axarr[1].set_ylim( [1e-4, 1e1] )
    axarr[1].set_ylabel(r'$w(\theta)$')
    axarr[1].set_xlabel(r'$\theta$')

    for i in range(njack):
        vfile = os.path.join(vdir, '%s.txt'%i)
        cfile = os.path.join(cdir, '%s.txt'%i)
        issfile = os.path.join(issdir, '%s.txt'%i)

        xi = np.loadtxt(vfile)
        coord = np.loadtxt(cfile)
        iss = np.loadtxt(issfile)

        axarr[1].plot(coord, xi, color=cmap[i])
        axarr[1].plot(coord, iss, color=cmap[i])



    on = 0
    files = []
    for j in range(njack):
        for ind in range(njack+1):
            if ind < njack:
                i = ind
            else:
                i = on

            vfile = os.path.join(vdir, '%s.txt'%i)
            cfile = os.path.join(cdir, '%s.txt'%i)
            issfile = os.path.join(issdir, '%s.txt'%i)

            xi = np.loadtxt(vfile)
            coord = np.loadtxt(cfile)
            iss = np.loadtxt(issfile)

            if i==on:
                if ind!=njack:
                    continue
                else:
                    axarr[2].plot(coord, xi, color=cmap[i])
                    axarr[2].plot(coord, iss, color=cmap[i])
                    axarr[0].plot(centers[i:(i+1), 0], centers[i:(i+1), 1], linestyle='None', marker='x', color='black', markersize=15)
            else:
                axarr[2].plot(coord, xi, 'gray', alpha=0.25)
                axarr[2].plot(coord, iss, 'gray', alpha=0.25)

            if not os.path.exists(outdir):
                os.makedirs(outdir)
    
        axarr[2].set_xscale('log')
        axarr[2].set_yscale('log')
        #axarr[2].set_ylim( [1e-4, 1e-1] )
        axarr[2].set_ylim( [1e-4, 1e1] )
        axarr[2].set_ylabel(r'$w(\theta)$')
        axarr[2].set_xlabel(r'$\theta$')

        file = os.path.join(outdir, '%i.png'%(on))
        files.append(file)
        fig.savefig(file)

        axarr[0].lines.pop(0)

        axarr[2].clear()
        on += 1

    #fstr = ' '.join(files)
    #outfile = os.path.join('all.pdf')
    #os.system('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=%s %s'%(outfile, fstr))

    #plt.show()
    sys.exit()

