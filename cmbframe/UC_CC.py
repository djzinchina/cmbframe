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

########################################################
#
#   UNIT CONVERSION AND COLOR CORRECTION CODE
#
#   Instruments supported: LFI(Average bandpass), 
#   HFI(Average bandpass), AliCPT(Tophat bandpass)
#   IRAS(Delta bandpass), WMAP(Delta bandpass)
#
#   Cross checked for HFI with few percent error.
#
#   Reference: Planck 2013 IX: HFI Spectral Response
#   
#   version 4: Shamik Ghosh, 2022-04-06 
#
########################################################## 
"""
This module computes unit conversion and color correction factors for CMB observations
based on Planck 2013 IX: HFI Spectral Response. This is meant to be a python equivalent
to the Planck UC_CC IDL codes.
"""
import numpy as np
import healpy as hp 
import astropy.io.fits as fits
import scipy.constants as con
from sys import exit

from . import em_law as el 

__pdoc__ = {}
__pdoc__['alicpt_tophat_bandpass'] = False
__pdoc__['get_normalized_transmission'] = False

T_CMB = 2.7255 # K

RIMO_path = '/home/doujzh/DATA/Planck_RIMO'

def alicpt_tophat_bandpass(band):
    if band == '95':
        nu_min_in_GHz = 77111500800. / con.giga
        nu_max_in_GHz = 112888504320. / con.giga
    elif band == '150':
        nu_min_in_GHz = 127769985024. / con.giga
        nu_max_in_GHz = 172230000640. / con.giga
    else:
        print("ERROR: Not a valid AliCPT band.")
        exit()
    
    nus_in_GHz = np.linspace(nu_min_in_GHz, nu_max_in_GHz, 100)
    bandpass = np.ones_like(nus_in_GHz)
    bandpass /= np.trapz(bandpass, x=nus_in_GHz*con.giga)

    return nus_in_GHz, bandpass

def get_normalized_transmission(inst, band):
    if inst =='LFI':
        RIMO_file = RIMO_path+'/LFI_RIMO_R3.31.fits'
        if not(band in ['030', '044', '070']):
            print('ERROR: Not a valid LFI band.')
            exit()

    if inst =='HFI':
        RIMO_file = RIMO_path+'/HFI_RIMO_R3.00.fits'
        if not(band in ['100', '143', '217', '353', '545', '857']):
            print('ERROR: Not a valid HFI band.')
            exit()

    hdul = fits.open(RIMO_file)
    if inst == 'LFI':
        nus_in_GHz = hdul['BANDPASS_'+band].data['WAVENUMBER']
        transmission = hdul['BANDPASS_'+band].data['TRANSMISSION']

        nus_in_GHz = nus_in_GHz[transmission>0.001*np.max(transmission)]
        transmission = transmission[transmission>0.001*np.max(transmission)]

        transmission /= np.trapz(transmission, x=nus_in_GHz*con.giga)

    elif inst == 'HFI':
        wavenumber_in_cminv = hdul['BANDPASS_F'+band].data['WAVENUMBER']
        transmission = hdul['BANDPASS_F'+band].data['TRANSMISSION']

        nus_in_GHz = (wavenumber_in_cminv * con.c / con.centi) / con.giga
        # print(nus_in_GHz, transmission)
        nus_in_GHz = nus_in_GHz[transmission>0.001*np.max(transmission)]
        transmission = transmission[transmission>0.001*np.max(transmission)]
        
        transmission /= np.trapz(transmission, x=nus_in_GHz*con.giga)
        # print(transmission, np.max(transmission))

    
    # nus_in_GHz = nus_in_GHz[transmission >= 1.e-6]
    # transmission = transmission[transmission >= 1.e-6]

    return nus_in_GHz, transmission

def bandpass_weights(inst, band):
    """
    This function fetches frequency in GHz and normalized bandpass weights for LFI, HFI and AliCPT.
    Currently the function fetches only the average bandpass for LFI and HFI, 
    and only tophat bandpass for AliCPT.

    Parameters
    ----------
    inst : {'LFI', 'HFI', 'AliCPT'}
        Instrument name. One of 'LFI', 'HFI', or 'AliCPT'
    band : {'030', '044', '070', '100', '143', '217', '353', '545', '857', '95', '150'}
        For LFI '030', '044' and '070' bands are supported. 
        For HFI '100', '143', '217', '353', '545' and '857' bands are supported.
        For AliCPT '95', '150' bands are supported  (Tophat bandpass assumed)  

    Returns
    -------
    nus_in_GHz : numpy 1D array
        A numpy 1D array of frequency in GHz over which the bandpass is normalized. 
        Note: Only use this frequency array for bandpass integration. Or bandpass 
        normalization may be wrong. 
    weights : numpy 1D array
        A numpy 1D array of bandpass weights for use in numerical bandpass integration
        over the frequency range given by nus_in_GHz.

    Notes
    -----
    If you are computing bandpass integration over wavenumber or wavelength then bandpass 
    normalization will be incorrect.
    """

    if inst in ['LFI', 'HFI']:
        return get_normalized_transmission(inst, band)
    elif inst == 'AliCPT':
        return alicpt_tophat_bandpass(band)
    else:
        print('ERROR: Only HFI, LFI and AliCPT supported currently.')
        exit()

def convert_KCMB_to_ySZ(inst, band):
    """
    Computes conversion factor from K_CMB to y SZ (Compton parameter). 

    Parameters
    ----------
    inst : {'WMAP', 'LFI', 'HFI', 'AliCPT', 'IRIS'}
        Instrument name: 'WMAP', 'LFI', 'HFI', 'AliCPT' or 'IRIS'
    band : {'K', '030', '044', '070', '100', '143', '217', '353', '545', '857', '95', '150', '3000'}
        For WMAP only 'K' band is supported presently (delta bandpass assumed)
        For LFI '030', '044' and '070' bands are supported
        For HFI '100', '143', '217', '353', '545' and '857' bands are supported
        For AliCPT '95', '150' bands are supported  (Tophat bandpass assumed)    
        For IRIS only '3000' (100 micron) band is supported presently (delta bandpass assumed)

    Returns
    -------
    float
        A float value that is the conversion factor from K_CMB to y SZ.
    """

    if inst != 'IRIS' and inst  != 'WMAP':
        nus_in_GHz, weights = bandpass_weights(inst, band)
        # print(nus_in_GHz)
        differential_bbody = el.B_prime_nu_T(nus_in_GHz)
        ysz = el.ysz_spectral_law(nus_in_GHz)

        band_integrated_CMB = np.trapz(weights*differential_bbody, x=nus_in_GHz * con.giga)
        band_integrated_ysz = np.trapz(weights*ysz, x=nus_in_GHz * con.giga)

    # print(nus_in_GHz)
# Referenece: Eq 33 from Planck 2013 XI HFI spectral response  
        return band_integrated_CMB/band_integrated_ysz  # returns in K_CMB^(-1) 
    elif inst == 'IRIS':
        nus_in_GHz = 3000. #in GHz for 100 micron

        differential_bbody = el.B_prime_nu_T(nus_in_GHz)
        ysz = el.ysz_spectral_law(nus_in_GHz)

        return differential_bbody / ysz 

    elif inst == 'WMAP':
        nus_in_GHz = 23. #in GHz for 100 micron

        differential_bbody = el.B_prime_nu_T(nus_in_GHz)
        ysz = el.ysz_spectral_law(nus_in_GHz)

        return differential_bbody / ysz 

def convert_KCMB_to_MJysr(inst, band):
    """
    Gives conversion factor from K_CMB to MJy/sr.
    MJy/sr assumes nu^-1 (IRAS) reference spectrum following Planck. 

    Parameters
    ----------
    inst : {'WMAP', 'LFI', 'HFI', 'AliCPT', 'IRIS'}
        Instrument name: 'WMAP', 'LFI', 'HFI', 'AliCPT' or 'IRIS'
    band : {'K', '030', '044', '070', '100', '143', '217', '353', '545', '857', '95', '150', '3000'}
        For WMAP only 'K' band is supported presently (delta bandpass assumed)
        For LFI '030', '044' and '070' bands are supported
        For HFI '100', '143', '217', '353', '545' and '857' bands are supported
        For AliCPT '95', '150' bands are supported  (Tophat bandpass assumed)    
        For IRIS only '3000' (100 micron) band is supported presently (delta bandpass assumed)

    Returns
    -------
    float
        A float value to convert K_CMB to MJy/sr.
    """

    LFI_bands = np.array(['030', '044', '070'])
    LFI_nucs  = np.array([28.4, 44.1, 70.4])

    HFI_bands = np.array(['100', '143', '217', '353', '545', '857'])
    HFI_nucs  = np.array([100., 143., 217., 353., 545., 857.,])

    AliCPT_bands = np.array(['95', '150'])
    AliCPT_nucs  = np.array([95., 150.])

    IRIS_bands = np.array(['3000'])
    IRIS_nucs  = np.array([3000.])

    WMAP_bands = np.array(['K'])
    WMAP_nucs = np.array([23.])

    if inst != 'IRIS' and inst != 'WMAP':
        nus_in_GHz, weights = bandpass_weights(inst, band)
        if inst == 'LFI':
            nuc_in_GHz = LFI_nucs[np.where(LFI_bands == band)[0]]
        elif inst == 'HFI':
            nuc_in_GHz = HFI_nucs[np.where(HFI_bands == band)[0]]
        elif inst == 'AliCPT':
            nuc_in_GHz = AliCPT_nucs[np.where(AliCPT_bands == band)[0]]

        differential_bbody = el.B_prime_nu_T(nus_in_GHz)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        band_integrated_CMB = np.trapz(weights*differential_bbody, x=nus_in_GHz * con.giga)
        band_integrated_nucbynu = np.trapz(weights*nuc_by_nu, x=nus_in_GHz * con.giga)

        return (band_integrated_CMB / band_integrated_nucbynu) * 1.e20  # 1.e20 factor converts W/m2/Hz to MJy.
    elif inst == 'IRIS':
        nus_in_GHz = 3000.
        nuc_in_GHz = IRIS_nucs[np.where(IRIS_bands == band)[0]][0]

        differential_bbody = el.B_prime_nu_T(nus_in_GHz)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        return (differential_bbody / nuc_by_nu) * 1.e20

    elif inst == 'WMAP':
        nus_in_GHz = 23.
        nuc_in_GHz = WMAP_nucs[np.where(WMAP_bands == band)[0]][0]

        differential_bbody = el.B_prime_nu_T(nus_in_GHz)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        return (differential_bbody / nuc_by_nu) * 1.e20

def convert_MJysr_to_Kb(inst, band):
    """
    Gives conversion factor from MJy/sr to brightness temperature Kb (K_RJ).

    Parameters
    ----------
    inst : {'WMAP', 'LFI', 'HFI', 'AliCPT', 'IRIS'}
        Instrument name: 'WMAP', 'LFI', 'HFI', 'AliCPT' or 'IRIS'
    band : {'K', '030', '044', '070', '100', '143', '217', '353', '545', '857', '95', '150', '3000'}
        For WMAP only 'K' band is supported presently
        For LFI '030', '044' and '070' bands are supported
        For HFI '100', '143', '217', '353', '545' and '857' bands are supported
        For AliCPT '95', '150' bands are supported    
        For IRIS only '3000' (100 micron) band is supported presently
    
    Notes
    -----
    No bandpass is required only central frequency is assumed.

    Returns
    -------
    float
        A float value to convert MJy/sr to Kb.
    """

    LFI_bands = np.array(['030', '044', '070'])
    LFI_nucs  = np.array([28.4, 44.1, 70.4])

    HFI_bands = np.array(['100', '143', '217', '353', '545', '857'])
    HFI_nucs  = np.array([100., 143., 217., 353., 545., 857.,])

    AliCPT_bands = np.array(['95', '150'])
    AliCPT_nucs  = np.array([95., 150.])

    IRIS_bands = np.array(['3000'])
    IRIS_nucs  = np.array([3000.])

    WMAP_bands = np.array(['K'])
    WMAP_nucs = np.array([23.])


    # nus_in_GHz, weights = bandpass_weights(inst, band)
    if inst == 'LFI':
        nuc_in_GHz = LFI_nucs[np.where(LFI_bands == band)[0]]
    elif inst == 'HFI':
        nuc_in_GHz = HFI_nucs[np.where(HFI_bands == band)[0]]
    elif inst == 'AliCPT':
        nuc_in_GHz = AliCPT_nucs[np.where(AliCPT_bands == band)[0]]
    elif inst == 'IRIS':
        nuc_in_GHz = IRIS_nucs[np.where(IRIS_bands == band)[0]][0]
    elif inst == 'WMAP':
        nuc_in_GHz = WMAP_nucs[np.where(WMAP_bands == band)[0]][0]

    return con.c**2. / 2. / (nuc_in_GHz * con.giga)**2. / con.k / 1.e20    # 1e20 is conversion factor from SI unit of emissivity to MJy 1e-6 x 1e26 = 1e20

def cc_dust_to_IRAS(inst, band, beta_d, T_d):
    """
    Function to compute band integrated colour correction factor to go from
    dust SED with spectral index of \(\\beta_d\) to \(\\nu_c/\\nu\) IRAS spectra in MJy/sr

    May be incorrect for WMAP if WMAP does not use IRAS reference. (To be checked...)

    Parameters
    ----------
    inst : {'WMAP', 'LFI', 'HFI', 'AliCPT', 'IRIS'}
        Instrument name: 'WMAP', 'LFI', 'HFI', 'AliCPT' or 'IRIS'
    band : {'K', '030', '044', '070', '100', '143', '217', '353', '545', '857', '95', '150', '3000'}
        For WMAP only 'K' band is supported presently (delta bandpass assumed)
        For LFI '030', '044' and '070' bands are supported
        For HFI '100', '143', '217', '353', '545' and '857' bands are supported
        For AliCPT '95', '150' bands are supported (tophat bandpass assumed)
        For IRIS only '3000' (100 micron) band is supported presently (delta bandpass assumed)
    beta_d : float
        dust spectral index
    T_d : float
        dust temperature

    Returns
    -------
    float
        A float value to convert dust reference spectrum to IRAS reference spectrum.
    """

    LFI_bands = np.array(['030', '044', '070'])
    LFI_nucs  = np.array([28.4, 44.1, 70.4])

    HFI_bands = np.array(['100', '143', '217', '353', '545', '857'])
    HFI_nucs  = np.array([100., 143., 217., 353., 545., 857.,])

    AliCPT_bands = np.array(['95', '150'])
    AliCPT_nucs  = np.array([95., 150.])

    IRIS_bands = np.array(['3000'])
    IRIS_nucs  = np.array([3000.])

    WMAP_bands = np.array(['K'])
    WMAP_nucs = np.array([23.])

    if inst != 'IRIS' and inst != 'WMAP':
        nus_in_GHz, weights = bandpass_weights(inst, band)
        if inst == 'LFI':
            nuc_in_GHz = LFI_nucs[np.where(LFI_bands == band)[0]]
        elif inst == 'HFI':
            nuc_in_GHz = HFI_nucs[np.where(HFI_bands == band)[0]]
        elif inst == 'AliCPT':
            nuc_in_GHz = AliCPT_nucs[np.where(AliCPT_bands == band)[0]]

        mod_bbody = el.greybody(nus_in_GHz, nuc_in_GHz, beta_d, T_d)#el.modified_blackbody(nus_in_GHz, beta_d, T_d) / el.modified_blackbody(nuc_in_GHz, beta_d, T_d)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        band_integrated_MBB = np.trapz(weights*mod_bbody, x=nus_in_GHz * con.giga)
        band_integrated_nucbynu = np.trapz(weights*nuc_by_nu, x=nus_in_GHz * con.giga)

        return (band_integrated_MBB / band_integrated_nucbynu) #/ 1.e20  # 1.e20 factor converts W/m2/Hz to MJy.
        # return (band_integrated_nucbynu / band_integrated_MBB)
    elif inst == 'IRIS':
        nus_in_GHz = 3000.
        nuc_in_GHz = IRIS_nucs[np.where(IRIS_bands == band)[0]][0]

        mod_bbody = el.greybody(nus_in_GHz, nuc_in_GHz, beta_d, T_d)#el.modified_blackbody(nus_in_GHz, beta_d, T_d) / el.modified_blackbody(nuc_in_GHz, beta_d, T_d)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        return (mod_bbody / nuc_by_nu) #/ 1.e20
        # return (nuc_by_nu / mod_bbody)

    elif inst == 'WMAP':
        nus_in_GHz = 23.
        nuc_in_GHz = WMAP_nucs[np.where(WMAP_bands == band)[0]][0]

        mod_bbody = el.greybody(nus_in_GHz, nuc_in_GHz, beta_d, T_d) #el.modified_blackbody(nus_in_GHz, beta_d, T_d) / el.modified_blackbody(nuc_in_GHz, beta_d, T_d)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        return (mod_bbody / nuc_by_nu) #/ 1.e20
        # return (nuc_by_nu / mod_bbody)

def cc_powerlaw_to_IRAS(inst, band, beta_s):
    """
    Computes band integrated colour correction factor to go from
    powerlaw SED with spectral index of \(\\beta_s\) to \(\\nu_c/\\nu\) IRAS spectra in MJy/sr

    May be incorrect for WMAP if WMAP does not use IRAS reference. (To be checked...)

    Parameters
    ----------
    inst : {'WMAP', 'LFI', 'HFI', 'AliCPT', 'IRIS'}
        Instrument name: 'WMAP', 'LFI', 'HFI', 'AliCPT' or 'IRIS'
    band : {'K', '030', '044', '070', '100', '143', '217', '353', '545', '857', '95', '150', '3000'}
        For WMAP only 'K' band is supported presently (delta bandpass assumed)
        For LFI '030', '044' and '070' bands are supported
        For HFI '100', '143', '217', '353', '545' and '857' bands are supported
        For AliCPT '95', '150' bands are supported (tophat bandpass assumed)
        For IRIS only '3000' (100 micron) band is supported presently (delta bandpass assumed)
    beta_s : float
        synchrotron spectral index

    Returns
    -------
    float
        A float value to convert synchrotron reference spectrum to IRAS reference spectrum.
    """

    LFI_bands = np.array(['030', '044', '070'])
    LFI_nucs  = np.array([28.4, 44.1, 70.4])

    HFI_bands = np.array(['100', '143', '217', '353', '545', '857'])
    HFI_nucs  = np.array([100., 143., 217., 353., 545., 857.,])

    AliCPT_bands = np.array(['95', '150'])
    AliCPT_nucs  = np.array([95., 150.])

    IRIS_bands = np.array(['3000'])
    IRIS_nucs  = np.array([3000.])

    WMAP_bands = np.array(['K'])
    WMAP_nucs = np.array([23.])

    if inst != 'IRIS' and inst != 'WMAP':
        nus_in_GHz, weights = bandpass_weights(inst, band)
        if inst == 'LFI':
            nuc_in_GHz = LFI_nucs[np.where(LFI_bands == band)[0]]
        elif inst == 'HFI':
            nuc_in_GHz = HFI_nucs[np.where(HFI_bands == band)[0]]
        elif inst == 'AliCPT':
            nuc_in_GHz = AliCPT_nucs[np.where(AliCPT_bands == band)[0]]

        power_law = el.powerlaw(nus_in_GHz, nuc_in_GHz, spec_ind=beta_s)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        band_integrated_pow = np.trapz(weights*power_law, x=nus_in_GHz * con.giga)
        band_integrated_nucbynu = np.trapz(weights*nuc_by_nu, x=nus_in_GHz * con.giga)

        return (band_integrated_pow / band_integrated_nucbynu) #/ 1.e20  # 1.e20 factor converts W/m2/Hz to MJy.
        # return (band_integrated_nucbynu / band_integrated_pow)
    elif inst == 'IRIS':
        nus_in_GHz = 3000.
        nuc_in_GHz = IRIS_nucs[np.where(IRIS_bands == band)[0]][0]

        power_law = el.powerlaw(nus_in_GHz, nuc_in_GHz, spec_ind=beta_s)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        return (power_law / nuc_by_nu) #/ 1.e20
        # return (nuc_by_nu / power_law)
    elif inst == 'WMAP':
        nus_in_GHz = 23.
        nuc_in_GHz = WMAP_nucs[np.where(WMAP_bands == band)[0]][0]

        power_law = el.powerlaw(nus_in_GHz, nuc_in_GHz, spec_ind=beta_s)
        nuc_by_nu = nuc_in_GHz / nus_in_GHz

        return (power_law / nuc_by_nu) #/ 1.e20
        # return (nuc_by_nu / power_law)