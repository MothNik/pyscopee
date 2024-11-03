# 💡〰️ `pyscopee` Changelog 〰️📝

The following symbols are used to indicate the state of the listed developments:

- ⚡️ Numba-accelerated version available (for `pip install pyscopee[fast]`)
- ✅ stable, unlikely to change
- 🚧 functionality as expected but future API changes planned
- 🧪 experimental feature

## Version 0.0b1

- Initial pre-release version

### New Features

- implementation of autoregressive extrapolation via the `augment.extrapolate` module:

  - a general autoregressive extrapolation function `.extrapolate_autoregressive` (⚡️✅)
  - an AR-coefficient regression function `.arburg` (⚡️✅) that is
    based on the segmented Burg algorithm with optional Tikhonov regularization

- experimental implementations of Mid-Infrared spectra simulations via the
  `spectra_simulate` module:
  - functions to simulate a Planck blackbody spectrum
    `.black_body_spectrum` (🧪) and its peak location
    `.black_body_peak` (🧪)
  - a function to simulate atmospheric absorption spectra via the HITRAN database
    and the `radis` package `.atmospheric_transmittance` (🧪🚧)
