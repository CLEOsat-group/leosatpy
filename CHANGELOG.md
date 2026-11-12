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
- **Trail quality checks.** A detected trail is rejected, and no magnitude is reported,
  if the TLE prediction does not put the satellite on the detector during the exposure,
  if its direction differs from the predicted one by more than `TRAIL_TLE_ANG_TOL`
  (default 5 deg), or if more than `TRAIL_MASK_FRAC_MAX` (default 50 %) of the trail
  aperture is masked. Found in an end-to-end run: the bleed column of a saturated star
  was taken as a trail, and 95 % of its aperture was masked (fill factor 0.036). New
  result columns: `QlfTrail`, `TrailTleAngDiff`.
- **Reference images.** Only exposures of the same pointing without a detected trail
  (in the same HDU) are used as reference images, as the documentation describes.
  Before, every other exposure was used, and a trail in the reference image could be
  detected as a source and mask a band of the trail image. Exposures without such a
  reference fall back to masking the catalogue stars, which leaves fainter stars
  along the trail unmasked (in the test night 0.03 mag for one trail). A reference
  image without the HDU of the trail image is skipped instead of failing.
- **Failed catalog downloads no longer stop a run.** Gaia and GSC downloads have a
  timeout per attempt (600 s) and are retried (3 attempts, 15 s apart). If all attempts
  fail, only the affected frame (calibration) or satellite (analysis) fails, with
  the reason in the log and the fails file, and the run continues. Before, one failed
  per-frame Gaia download ended the calibration run, an unreachable GSC service
  ended the analysis run (`sys.exit`), and a server that never answered hung the run.
  A GSC cone without stars no longer raises an error.
- A saturation limit taken from a header keyword no longer fails.
- Reduced images and all later products (calibrated images, catalogs, figures, the
  `File` column of the result table) get clean names: spaces become `_` and brackets
  and similar characters are removed (`STARLINK-11190 [DTC]-0001` →
  `STARLINK-11190_DTC-0001`). The raw name is kept in the `RAWFILE` header keyword;
  the raw files are not renamed. A reduced image written under the old name is
  replaced. Entries in the ROI, glint-mask and faint-trail tables must use the clean
  names for data reduced with 0.7.0.
- Figures whose file names contain spaces or brackets (e.g. `STARLINK-11190 [DTC]-0001`)
  are trimmed correctly. `mogrify` and `pdfcrop` were called through the shell with an
  unquoted path, which split such names and would have run shell characters in a file
  name; they are now called without a shell, and a missing tool only skips the trimming.
- The trail aperture correction no longer becomes `-inf` when the flux ratio is not positive.

### Changed
- Gaia downloads are limited to the brightest `REF_CAT_MAX_ROWS` stars per cone
  (default 5000). Dense fields near the Galactic plane were slow and could fail with
  archive errors (HTTP 500) when downloading up to 50000 stars to G = 19. Cones that
  reach the limit are cached as complete only to their faintest star; cones cached by
  earlier versions remain valid.
- The Gaia prefetch reports how many cones are already cached and how many will be
  downloaded, and shows `[i/N]` progress.
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
