# Changelog

## 0.7.0

Magnitudes from this version differ from those of 0.6.0 and earlier. Do not mix
result tables from the two versions.

### Fixed (these bias the magnitudes)
- **Sign of the stellar aperture correction.** The zero point comes from the stellar
  fluxes in the finite aperture, so the stellar aperture correction
  `MagCorrect = -2.5 log10(F_inf / F_aper)` (negative) has to be *subtracted*:
  `m = -2.5 log10(F_sat) + ZP_aper - MagCorrect + MagCorrectSat`.
  Up to 0.6.0 it was added, which made all satellite magnitudes too bright by
  `2 * |MagCorrect|` (typically 0.02-0.2 mag). Old tables (both `zp` and `legacy`
  modes, not the header zero-point columns `*_zp`) can be corrected approximately with
  `Mag_new = Mag_old - 2 * MagCorrect`. The other changes below are not included in that
  correction.
- **Trail flux under masked pixels.** Pixels masked inside the trail aperture (stars
  along the trail, saturated or bad pixels) were missing from the trail flux, so the
  trail came out too faint. The flux is now divided by a profile-weighted fill factor.
  New result columns: `TrailMaskFrac`, `TrailFillFactor`. A warning is logged when the
  trail core is masked along its full length (the flux is then a lower limit).
- **Image units.** Images that were already in electrons (reduction with
  `CORRECT_GAIN = True`, the default) were labelled as ADU and multiplied by the gain a
  second time in the analysis. That shifted the saturation mask (to 1/gain of the true
  limit) and the Poisson term of the flux errors. The unit is now read from `BUNIT`
  (`adu`, `counts`, `electron`, `e-`, ...), and images are converted to electrons once.
  Background files written before 0.7.0 are re-estimated automatically.
- A saturation limit taken from a header keyword no longer fails.
- The trail aperture correction no longer becomes `-inf` when the flux ratio is not positive.

### Changed
- Figures show the actual image unit (e-) instead of "counts".
- The S/N used to choose the optimum aperture is now the flux divided by its measured
  error, so it includes the sky noise from the annulus. Before, the sky term was about
  zero on the background-subtracted image. The sky annulus stays outside the aperture
  for large radii.
- Sky annuli use the sigma-clipped (3 sigma) median instead of the mean.
- The optimum and the large aperture now share one sky annulus outside the large
  aperture, for both stars and trails.
- The stellar aperture correction is derived from stars with S/N >= 50 (or the
  brightest quarter), and `e_MagCorrect` is the standard error of the median instead of
  the scatter of all stars.
- `FluxScale < 1` (impossible, since the satellite cannot be on the detector longer than
  the exposure) is set to 1 with a warning.
- The arbitrary 5 % pixel-scale uncertainty was removed from the trail length errors.
- Corrected the error propagation of the SDSS -> BVRI (Lupton 2005) and U (Jester 2005)
  transformations.
