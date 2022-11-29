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
CMBframe is a CMB data analysis framework developed for use in the AliCPT project.
"""
__pdoc__ = {}
__pdoc__['cmbframe.version'] = False
__pdoc__['cmbframe.super_pix'] = False
__pdoc__['cmbframe.group_ajacent_pix'] = False
__pdoc__['cmbframe.covarifast'] = False

import warnings
warnings.simplefilter("ignore")


from .version import __version__ 

from .cleaning_utils import (
    compute_beam_ratio,
    beam_ratios,
    calc_binned_cov,
    compute_Ncov
)

from .plot_utilts import (
    make_plotaxes,
    plot_ilc_weights,
    plot_gls_weights,
    plot_needlet_bands,
    plot_maps,
    plot_needlet_maps
)

from .template_cleaner import (
    Emode_recycler,
    get_cleanedBmap,
    get_residual,
    get_leakage
)

from .wavelet import (
    get_lmax_band,
    cosine_bands,
    gaussian_bands,
    alm2wavelet,
    wavelet2alm, 
    map2wavelet,
    wavelet2map,
    write_waveletmap_fits,
    read_waveletmap_fits,
    read_waveletbands_fits

)

from .nmt_wrapper import (
    setup_bins, 
    binner, 
    bin_error, 
    map2coupCl_nmt, 
    map2Cl_nmt
)

from .hp_wrapper import (
    iqu2teb, 
    calc_binned_Cl,
    roll_bin_Cl,
    harmonic_udgrade,
    mask_udgrade,
    alm_fort2c,
    alm_c2fort
)

from .UC_CC import (
    bandpass_weights,
    convert_KCMB_to_ySZ,
    convert_KCMB_to_MJysr,
    convert_MJysr_to_Kb,
    cc_dust_to_IRAS, 
    cc_powerlaw_to_IRAS
)

from .em_law import (
    B_nu_T,
    B_prime_nu_T,
    powerlaw,
    ysz_spectral_law,
    greybody,
    modified_blackbody
)

#from .nilc import nilc_workspace
from .hilc import hilc_cleaner
from .constrained_ilc import cilc_cleaner
from .gls_toolkit import gls_solver
# from .nilc_weights import nilc_weights_wsp_himem, nilc_weights_wsp_slo, get_projected_wav

from .Planck_colors import gaussian, foreground

from .outlier import get_filteredmap, outlier_cut
from .border_finder import nwhere, get_mask_border
from .inpaint_holes import inpaint_holes

from .wmap_noise import get_wmap_noise
from .alicpt_noise import get_alicpt_noise



