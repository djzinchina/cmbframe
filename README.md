# CMBframe 

#### *Cosmic Microwave Background (data analysis) frame(work)*

CMBframe has been developed for application in the Ali CMB Polarization Telescope (AliCPT) data analysis pipeline. It features set of classes, functions and subroutines for component separation of CMB temperature and polarization data, with fast wrappers for healpix and NaMaster routines and plotting with matplotlib.

Limited documentation now available at: https://1cosmologist.github.io/CMBframe/index.html or in the docs folder.

###### CHANGELOG
2022-05-23 : version 0.4.0
1. Limited documentation support now available. Will be gradually expanded.
2. Added new and improved NILC module to the package however, it is currently broken due to covarifast Fortran module linking issue. Will be fixed in later version.

2022-02-01 : version 0.3.1
1. Major bug fix in UC_CC

2022-01-28 : version 0.3.0
1. Added unit conversion and color correction functions.
   (Supports:bandpass_weights, convert_KCMB_to_ySZ,
    convert_KCMB_to_MJysr, convert_MJysr_to_Kb,
    cc_dust_to_IRAS, cc_powerlaw_to_IRAS)
2. Added emission law functions.
   (B_nu_T, B_prime_nu_T, powerlaw, ysz_spectral_law,
    greybody, modified_blackbody)

2022-01-14 : version 0.2.11
1. Added frequency band option for outlier filter

2021-12-03 : version 0.2.10
1. Added option for external seed in AliCPT/WMAP noise modules.
2. Bug fixes.

2021-11-30 : version 0.2.9
1. Added wide scan noise option for AliCPT noise.

2021-10-23 : version 0.2.8
1. Added conversion between C type and Fortran type indexing of healpy alm arrays.

2021-10-19 : version 0.2.7
1. Added I/O from fits files for wavelet transformed maps, based on astropy.
2. General bug fixes and optimizations.

versions 0.2.1 - 0.2.6
1. Various bug fixes

2021-07-08 : version 0.2.0
1. Added harmonic_degrade routine to hp_wrapper
2. Added mask_degrade routine to hp_wrapper
3. Added inpaint_holes inpainting function based on local surface fitting
4. Added outlier get_filteredmap and outlier_cut for points source mask preparation
5. Added faster super_pix2 module to replace super_pix neighbour calculation in nilc_weights
6. Added get_alicpt_noise and get_wmap_noise functions
7. General bug fixes. 
