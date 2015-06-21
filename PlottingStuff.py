import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import Utils


def VPlot(band='i'):
    #ashley = np.loadtxt('wxinzDESnorm.dat')
    ashley= np.loadtxt('wxinzDESnorm1.5.dat')
    outdir = 'CorrFiles'

    outlabel = 'v0b-z-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB0 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'v1b-z-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB1 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'v2b-z-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB2 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'v3b-z-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB3 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'bench-selection-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    mpts = np.loadtxt('McCracken/cosmos23-24.txt')
    spts = np.loadtxt('McCracken/subaru23-24.txt')
    cpts = np.loadtxt('McCracken/cfhtls23-24.txt')

    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    axarr = []
    axarr.append( fig.add_subplot(gs[0]) )

    axarr[0].errorbar( DB0[2], DB0[0], yerr=np.sqrt(np.diag(DB0[1])), color='red', fmt='o', label='DB v2')
    axarr[0].errorbar( DB1[2], DB1[0], yerr=np.sqrt(np.diag(DB1[1])), color='blue', fmt='o', label='DB v3')
    axarr[0].errorbar( DB2[2], DB2[0], yerr=np.sqrt(np.diag(DB2[1])), color='yellow', fmt='o', label='DB v3_2')
    axarr[0].errorbar( DB3[2], DB3[0], yerr=np.sqrt(np.diag(DB3[1])), color='green', fmt='o', label='DB v3_3')
    
    axarr[0].errorbar( DB[2], DB[0], yerr=np.sqrt(np.diag(DB[1])), color='orange', fmt='o', label='DB Full')
    axarr[0].plot(ashley[:,0], ashley[:,1], color='black', ls='--', label="Ashley's Prediction")

    axarr[0].errorbar( mpts[:,0], mpts[:,1], yerr=mpts[:,3], color='black', fmt='o', markersize=6, label='ACS')
    #axarr[0].errorbar( spts[:,0], spts[:,1], yerr=spts[:,3], color='pink', fmt='s', markersize=6, label='Subaru')
    #axarr[0].errorbar( cpts[:,0], cpts[:,1], yerr=cpts[:,3], color='cyan', fmt='^', markersize=6, label='CFHTLS')

    axarr[0].set_xscale('log')
    axarr[0].legend(loc='best')
    axarr[0].set_xlabel(r'$\theta$ [deg]')
    #axarr[0].set_ylabel(r'$w_{BB}(\theta)$')
    axarr[0].set_ylabel(r'$w(\theta)$')
    axarr[0].set_yscale('log')

    axarr[0].set_ylim( [1e-4, 1] )
    plt.show()


def PlotOne(dir, outdir='CorrFiles', band='i', kind='DB', plotkwargs={}, ax=None, contam=0, get=False, noplot=False):
    #dir = 'BM-no%i-23-24'

    if dir=='ACS':
        db = np.loadtxt('McCracken/cosmos23-24.txt')
        DB = [ db[:,1], np.diag(db[:,3]*db[:,3]), db[:,0] ]
    elif dir=='ACS21':
        db = np.loadtxt('McCracken/cosmos21-22.txt')
        DB = [ db[:,1], np.diag(db[:,3]*db[:,3]), db[:,0] ]

    elif dir=='ASHLEY':
        db = np.loadtxt('wxinzDESnorm.dat')
    elif dir=='a15':
        db = np.loadtxt('wxinzDESnorm1.5.dat')
    elif dir=='a17':
        db = np.loadtxt('wxinzDESnorm1.7.dat')
    elif dir=='a2':
        db = np.loadtxt('wxinzDESnormg2.dat')

    else:
        basepath = os.path.join(outdir, dir, band)
        DB = Utils.GetVecInfo(os.path.join(basepath, kind))
        DB[0] = DB[0] * np.power(1 + contam, 2.0)

    if ax is None:
        fig = plt.figure(figsize=(8, 6), tight_layout=True)
        gs = gridspec.GridSpec(1,1)
        axarr = []
        axarr.append( fig.add_subplot(gs[0]) )
        ax = axarr[0]


    if not noplot:
        if dir in ['ASHLEY', 'a15', 'a17', 'a2']:
            ax.plot(db[:,0], db[:,1], **plotkwargs)
        else:
            ax.errorbar( DB[2], DB[0], yerr=np.sqrt(np.diag(DB[1])), **plotkwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim([1e-4, 1])
        ax.set_xlim([3e-4, 10])

        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$w(\theta)$')
        
        ax.grid(b=False, which='minor')

    if get:
        return DB[2], DB[0], DB[1]



def VPlot2(band='i'):
    #ashley = np.loadtxt('wxinzDESnorm.dat')
    ashley15 = np.loadtxt('wxinzDESnorm1.5.dat')
    ashley17 = np.loadtxt('wxinzDESnorm1.7.dat')
    ashley20 = np.loadtxt('wxinzDESnormg2.dat')
    outdir = 'CorrFiles'

    outlabel = 'bench-selection-bugfix-v0-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB0 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'bench-selection-bugfix-v1-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB1 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'bench-selection-bugfix-v2-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB2 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    outlabel = 'bench-selection-bugfix-v3-23-24'
    #outlabel = 'bench-selection-bugfix-v3-23-24-test'
    basepath = os.path.join(outdir, outlabel, band)
    DB3 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    #outlabel = 'bench-selection-bugfix-23-24'
    outlabel = 'bench-selection-bugfix-extracuts-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB = Utils.GetVecInfo(os.path.join(basepath, 'DB'))

    mpts = np.loadtxt('McCracken/cosmos23-24.txt')

    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    axarr = []
    axarr.append( fig.add_subplot(gs[0]) )

    axarr[0].plot(ashley15[:,0], ashley15[:,1], color='black', ls='-.', label="Ashley1.5")
    axarr[0].plot(ashley17[:,0], ashley17[:,1], color='black', ls='--', label="Ashley1.7")
    axarr[0].plot(ashley20[:,0], ashley20[:,1], color='black', ls='-', label="Ashley2.0")

    axarr[0].errorbar( DB0[2], DB0[0], yerr=np.sqrt(np.diag(DB0[1])), color='red', fmt='o', label='DB v2')
    axarr[0].errorbar( DB1[2], DB1[0], yerr=np.sqrt(np.diag(DB1[1])), color='blue', fmt='o', label='DB v3')
    axarr[0].errorbar( DB2[2], DB2[0], yerr=np.sqrt(np.diag(DB2[1])), color='green', fmt='o', label='DB v3_2')
    axarr[0].errorbar( DB3[2], DB3[0], yerr=np.sqrt(np.diag(DB3[1])), color='yellow', fmt='o', label='DB v3_3')
    
    axarr[0].errorbar( DB[2], DB[0], yerr=np.sqrt(np.diag(DB[1])), color='orange', fmt='o', label='DB Full')
    axarr[0].errorbar( mpts[:,0], mpts[:,1], yerr=mpts[:,3], color='black', fmt='o', markersize=6, label='ACS')

    axarr[0].set_xscale('log')
    axarr[0].legend(loc='best')
    axarr[0].set_xlabel(r'$\theta$ [deg]')
    #axarr[0].set_ylabel(r'$w_{BB}(\theta)$')
    axarr[0].set_ylabel(r'$w(\theta)$')
    axarr[0].set_yscale('log')

    axarr[0].set_ylim( [1e-4, 1] )
    plt.show()


def DPlot():
    ashley = np.loadtxt('wxinzDESnorm.dat')
    outdir = 'CorrFiles'


    outlabel = 'dec_-49_-40-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB0 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))
    DU0 = Utils.GetVecInfo(os.path.join(basepath, 'DU'))

    outlabel = 'dec_-54_-49-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB1 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))
    DU1 = Utils.GetVecInfo(os.path.join(basepath, 'DU'))

    outlabel = 'dec_-65_-54-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB2 = Utils.GetVecInfo(os.path.join(basepath, 'DB'))
    DU2 = Utils.GetVecInfo(os.path.join(basepath, 'DU'))

    outlabel = 'bench-selection-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB = Utils.GetVecInfo(os.path.join(basepath, 'DB'))
    DU = Utils.GetVecInfo(os.path.join(basepath, 'DU'))

    mpts = np.loadtxt('McCracken/cosmos23-24.txt')

    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    axarr = []
    axarr.append( fig.add_subplot(gs[0]) )

    axarr[0].errorbar( DB0[2], DB0[0], yerr=np.sqrt(np.diag(DB0[1])), color='red', fmt='o', label='DB dec1')
    axarr[0].errorbar( DB1[2], DB1[0], yerr=np.sqrt(np.diag(DB1[1])), color='blue', fmt='o', label='DB dec2')
    axarr[0].errorbar( DB2[2], DB2[0], yerr=np.sqrt(np.diag(DB2[1])), color='green', fmt='o', label='DB dec3')
    axarr[0].errorbar( DB[2], DB[0], yerr=np.sqrt(np.diag(DB[1])), color='orange', fmt='o', label='DB Full')
    axarr[0].errorbar( mpts[:,0], mpts[:,1], yerr=mpts[:,3], color='black', fmt='o', label='ACS')

    axarr[0].plot( DU0[2], DU0[0], color='red')
    axarr[0].plot( DU1[2], DU1[0], color='blue')
    axarr[0].plot( DU2[2], DU2[0], color='green')
    axarr[0].plot( DU[2], DU[0], color='orange',)

    axarr[0].plot(ashley[:,0], ashley[:,1], color='black', ls='--', label="Ashley's Prediction")

    axarr[0].set_xscale('log')
    axarr[0].legend(loc='best')
    axarr[0].set_xlabel(r'$\theta$ [deg]')
    #axarr[0].set_ylabel(r'$w_{BB}(\theta)$')
    axarr[0].set_ylabel(r'$w(\theta)$')
    axarr[0].set_yscale('log')

    plt.show()



def PlotCross():
    outdir = 'CorrFiles'
    outlabel = 'bench-selection-23-24'
    basepath = os.path.join(outdir, outlabel, band)
    DB = Utils.GetVecInfo(os.path.join(basepath, 'DxB'))

    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    axarr = []
    axarr.append( fig.add_subplot(gs[0]) )

    axarr[0].errorbar( DB[2], DB[0], yerr=np.sqrt(np.diag(DB[1])), color='red', fmt='o', label='DxB')
    axarr[0].set_xscale('log')
    axarr[0].legend(loc='best')
    axarr[0].set_xlabel(r'$\theta$ [deg]')
    #axarr[0].set_ylabel(r'$w_{BB}(\theta)$')
    axarr[0].set_ylabel(r'$w(\theta)$')
    axarr[0].set_yscale('log')

    plt.show()

