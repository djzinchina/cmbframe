#######################################################################
# This file is a part of CMBframe
#
# Cosmic Microwave Background (data analysis) frame(work)
# Copyright (C) 2021  Shamik Ghosh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information about CMBframe please visit 
# <https://github.com/1cosmologist/CMBframe> or contact Shamik Ghosh 
# at shamik@ustc.edu.cn
#
#########################################################################
"""
Module contains higher level matplotlib and healpy plot wrappers for common 
plotting requirements in CMB data analysis.
"""
import numpy as np 
import healpy as hp
import math as mt
import matplotlib.pyplot as plt 

from . import Planck_colors as pc

# Matplotlib settings
plt.style.use('seaborn-ticks')
plt.rcParams['font.sans-serif'] = 'Helvetica'
screen_dpi = 94
print_dpi = 600

def make_plotaxes(res='screen', shape='rec', need_axis=True):
    """
    Function produces new figure and axes object for a matplotlib plot with
    preset resolution and size options.

    Parameters
    ----------
    res : {'screen', 'print'}, optional
        Resolution of plot set by either for screen viewing or for publication
        quality graphics. Default is 'screen' with dpi of 94. while 'print' sets
        a dpi of 600.
    shape : {'rec', 'sq'}, optional
        Aspect ratio of the plot. Default is 'rec' for a rectangular axes with 
        1000 x 626 pixel or 3.5 x 2.16 inch size. If 'sq' is set it selects a
        nearly square aspect ratio with 600 x 626 pixel size or 3 x 3.2 inch size.
    need_axis : bool, optional
        Returns ``matplotlib`` axes with figure number and figure object. Default 
        is True.

    Returns
    -------
    fno : int
        The integer value of the current figure number.
    fig : matplotlib figure
        A new ``matplotlib figure`` object.
    ax : matplotlib axes, optional
        A new ``matplotlib axes`` object. 
    """
    if res == 'screen':
        my_dpi = screen_dpi
        if shape == 'rec':
            x_size = 1000/my_dpi
            y_size = 626/my_dpi
        elif shape == 'sq':
            x_size = 600/my_dpi
            y_size = 626/my_dpi
        plt.rc('font', family='sans-serif', size=16)

    elif res == 'print':
        my_dpi = print_dpi
        plt.rc('font', family='sans-serif', size=10)
        if shape == 'rec':
            x_size = 3.5
            y_size = 2.16
        elif shape == 'sq':
            x_size = 3.
            y_size = 3.2

    if plt.fignum_exists(1):
        fig_no = plt.gcf().number+1 
    else:
        fig_no = 1 

    if need_axis:
        fig_new, ax_new = plt.subplots(num=fig_no, figsize=(x_size, y_size), dpi=my_dpi)
        return fig_no, fig_new, ax_new
    else:
        fig_new = plt.figure(num=fig_no, figsize=(x_size, y_size), dpi=my_dpi)
        return fig_no, fig_new 

def plot_ilc_weights(wgts_in, wgt_type, label_list=None, outfile=None, resol='screen', show=True):
    
    nu_dim = len(wgts_in[:,0])
    lmax = len(wgts_in[0,:]) - 1

    ells = np.arange(lmax+1)

    if isinstance(label_list,list):
        use_label = label_list
    else : 
        use_label = list('band '+str(nu) for nu in range(nu_dim))

    fno, fig, ax = make_plotaxes(res=resol, shape='rec')

    for nu in range(nu_dim):
        ax.plot(ells[2:], wgts_in[nu,2:], label=use_label[nu])
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$w^\nu_\ell$ '+wgt_type)
    if resol =='screen': ax.legend(loc='best', frameon=False, fontsize=12)
    if resol =='print': ax.legend(loc='best', frameon=False, fontsize=8)
    ax.set_xscale("log")

    if outfile != None : plt.savefig(outfile,bbox_inches='tight',pad_inches=0.1)
    if show: plt.show()

def plot_gls_weights(wgts_in, comp, label_list=None, outfile=None, resol='screen', show=True):
    
    nu_dim = len(wgts_in[0,:,0])
    lmax = len(wgts_in[0,0,:]) - 1

    ells = np.arange(lmax+1)

    if isinstance(label_list,list):
        use_label = label_list
    else : 
        use_label = list('band '+str(nu) for nu in range(nu_dim))

    if comp == 0:
        wgt_type = 'CMB'
    elif comp == 1:
        wgt_type = 'Synchrotron'
    elif comp == 2:
        wgt_type = 'Dust'

    fno, fig, ax = make_plotaxes(res=resol, shape='rec')

    for nu in range(nu_dim):
        ax.plot(ells[2:], wgts_in[comp, nu, 2:], label=use_label[nu])
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$w^\nu_\ell$ '+wgt_type)
    if resol == 'screen': ax.legend(loc='best', frameon=False, fontsize=12)
    if resol == 'print': ax.legend(loc='best', frameon=False, fontsize=8)
    ax.set_xscale("log")
    if outfile != None : plt.savefig(outfile,bbox_inches='tight',pad_inches=0.1)

    if show: plt.show()

def plot_needlet_bands(bands_in, file_out=None, xscale=None, resol='screen', show=True):

    nbands = len(bands_in[0,:])
    lmax = len(bands_in[:,0])

    ells = np.arange(lmax)

    fno, fig, ax = make_plotaxes(res=resol, shape='rec')
    for band in range(nbands):
        ax.plot(ells, bands_in[:,band], lw=1.4, alpha=0.7) 
    ax.plot(ells, np.sum(bands_in**2., axis=1), 'k-', lw=1.7, alpha=0.5)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(r'$\ell$')

    if file_out !=None: plt.savefig(file_out, bbox_inches='tight', pad_inches=0.1)
    if show: plt.show()


def plot_maps(map_in, mask_in=None, title='', proj='moll', unit='', vmin=None, vmax=None, rot=[175., 50., 0.], outfile=None, norm=None, resol='screen', col='gauss', dgrid=30., show=True):
    if col == 'gauss':
        colmap = pc.gaussian
    elif col == 'fg':
        colmap = pc.foreground
    elif col == 'seq':
        colmap = plt.cm.plasma

    if isinstance(mask_in, (list, np.ndarray)):
        map_plot = (np.copy(map_in) * mask_in) + ((1 - mask_in) * hp.UNSEEN)
    else:
        map_plot = np.copy(map_in)

    if proj == 'moll':
        fno, fig = make_plotaxes(res=resol, shape='rec', need_axis=False)
        hp.mollview(map_plot, cmap=colmap, min=vmin, max=vmax, title=title, unit=unit, fig=fno, norm=norm)
    elif proj == 'orth':
        fno, fig = make_plotaxes(res=resol, shape='sq', need_axis=False)
        hp.orthview(map_plot, half_sky=True, rot=rot, cmap=colmap, unit=unit, min=vmin, max=vmax, title=title, norm=norm, fig=fno) #bcwor.bcwor
    else:
        print("ERROR: Only mollweide (moll) or orthographic (orth) projections available.")
        exit()

    hp.graticule(dpar=dgrid, dmer=dgrid, ls='-', lw=0.1)
        
    if outfile != None: plt.savefig(outfile, bbox_inches='tight',pad_inches=0.1)

    if show: plt.show()


def plot_needlet_maps(wvlt_maps, proj='moll', rot=[175., 50., 0.], unit='', outfile=None, col='gauss', show=True):
    plt.rc('font', family='sans-serif', size=10)
    my_dpi = 94

    if col == 'gauss':
        colmap = pc.gaussian
    elif col == 'fg':
        colmap = pc.foreground
    elif col == 'seq':
        colmap = plt.cm.plasma

    nbands = len(wvlt_maps)

    if proj == 'moll':
        ncol = 2 
    else :
        ncol = 3

    if plt.fignum_exists(1):
        fno = plt.gcf().number+1 
    else:
        fno = 1

    if proj == 'moll':
        nrow = mt.ceil(nbands/ncol)
        fig = plt.figure(num=fno, figsize=[1080/my_dpi, (nrow*480)/my_dpi], dpi=my_dpi)
    elif proj == 'orth':
        nrow = mt.ceil(nbands/ncol)
        fig = plt.figure(num=fno, figsize=[1080/my_dpi, (nrow*540)/my_dpi], dpi=my_dpi)
    else:
        print("ERROR: Projection only Mollweide (moll) or half-sky Orthographic (orth)")
        exit()

    for band in range(nbands):
        if proj == 'moll': 
            hp.mollview(wvlt_maps[band], fig=fno, unit=unit, sub=[nrow, ncol, band+1], cmap=colmap, title='Band '+str(band+1))
        elif proj == 'orth':
            hp.orthview(wvlt_maps[band], half_sky=True, rot=rot, fig=fno, unit=unit, sub=[nrow, ncol, band+1], cmap=colmap, title='Band '+str(band+1))
        else : 
            print("ERROR: Projection only Mollweide (moll) or half-sky Orthographic (orth)")
            exit()

    if outfile != None: plt.savefig(outfile,bbox_inches='tight',pad_inches=0.1)

    if show: plt.show()