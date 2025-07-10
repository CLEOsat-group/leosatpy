Usage instructions
==================

- :ref:`Running LEOSatpy`
- :ref:`Reduction in a nutshell`
- :ref:`WCS calibration in a nutshell`
- :ref:`Satellite trail analysis in a nutshell`

Running LEOSatpy
----------------

Analysing satellite data with LEOSatpy is straightforward and only requires a few commands to be entered in the terminal.
Upon the first-time execution, a copy of the configuration file :file:`leosatpy_config.ini` is placed in the ``/home/username/`` directory.
The parameters in the configuration file can be adjusted with a text editor (see :ref:`Configuration file`).

LEOSatpy offers some degree of freedom in the nomenclature and structuring of the folder.
However, it is recommended to follow the folder layout given below:

::

    .
    └── Telescope-Identifier <- free naming
        ├── YYYY-MM-DD <- recommended format
        │   ├── bias
        │   ├── flats
        │   ├── darks
        │   └── science_data <- free naming
        │       └── raw <- optional, but recommended
        ├── YYYY-MM-DD
        └── YYYY-MM-DD

.. attention::
   The main folder name should contain the date of observation in one of these formats:
   ``YYYY-MM-DD`` or ``YYYYMMDD``.

LEOSatpy automatically selects the search path for calibration data based on the observation date
from the science FITS-header and the folder names in the input path.
Valid formats include: ``20221110``, ``2022-11-20``, ``tel_20221011_satxy``, ``2022-11-26_satxy_obs1``.

.. tip::

   It is recommended to separate raw calibration files (bias, darks, and flats) from science observation files
   into their respective directories: :file:`bias/`, :file:`darks/`, :file:`flats/`, and :file:`science_data/raw/`.

LEOSatpy supports the following input options for running the software:

* ``-s`` or ``--silent``: Suppress most console output. Useful for batch processing.
* ``-v`` or ``--verbose``: Enable detailed logging for debugging.
* ``-w`` or ``--ignore_warnings``: Show all warnings about FITS headers. Default is ``True``.
* ``--version``: Display the current version of LEOSatpy.
* ``-h`` or ``--help``: Display help information for the command.

Once the data is organized, LEOSatpy is ready to use.

Reduction in a nutshell
-----------------------

The reduction of all raw FITS-files in a folder can be performed via the following line:
::

    reduceSatObs PATH/TO/TELESCOPE/DATA/NIGHT_X

LEOSatpy also accepts multiple input paths:
::

    reduceSatObs PATH/TO/DATA/NIGHT_X PATH/TO/DATA/NIGHT_Y

It is also possible to reduce all data from a telescope at once:
::

    reduceSatObs PATH/TO/TELESCOPE

.. note::
   Relative paths are also acceptable, e.g., ``reduceSatObs ../Telescope-Identifier/YYYY-MM-DD/``.

.. attention::
   To prevent unexpected behavior, please check that:

   * Raw FITS-files contain valid data, e.g., they are not empty or corrupted
   * FITS-header keywords (e.g., ``IMAGETYP`` for bias, flats, or science files) are correctly labeled
   * Corresponding raw calibration images are available with matching parameters (binning, exposure time, filter)

.. tip::
    The usage of partial and multiple inputs as shown above also works for the other programs in the package.

.. important::
   LEOSatpy will never overwrite original data.

.. attention::
   To ensure a successful reduction, it is recommended to check the raw calibration files in the ``bias``, ``flats``, and ``darks`` subdirectories before running the reduction.
   For example, check that the provided flats are not overexposed, i.e., that the mean flux value is well within the linearity limits of the detector.

The `reduceSatObs` command supports the following input options:

* ``-f`` or ``--force-reduction``: Force recalculation of master calibration files, even if they already exist.

During image reduction, LEOSatpy first searches the input path for available FITS-files, identifies the science data, and collects essential information, such as the telescope, observation date, and instrument settings from the FITS-header keywords.
Based on this information, LEOSatpy selects the appropriate calibration files (bias, darks, and flats), and copies them to a temporary directory for processing.

Each set of calibration files is then processed using `Ccdproc <https://ccdproc.readthedocs.io/en/latest/>`_ to create master calibration files, such as master bias, master dark, or master flat.
Which master calibration files are created ultimately depends on the available raw calibration files. However, if raw calibration files are available,
LEOSatpy allows the user to control which master calibration files are created (see :ref:`Reduction options <Reduction options>`), and how they are combined (see e.g., :ref:`combine_method_flat <Reduction options>`).
All master calibration files are saved in the ``master_calibs`` subdirectory.

The raw science FITS-files are then processed individually by applying the master calibration files to remove instrumental signatures and create the reduced FITS-image.
The reduced FITS-files are stored in the ``reduced`` subdirectory, and the associated information stored in the result table (see :ref:`result_table_name <General options>` in the configuration file).

Finally, LEOSatpy cleans up the temporary files used during the reduction process.

.. important::
   When finished, check the FITS-files that LEOSatpy writes to the subdirectories ``master_calibs`` and ``reduced``.

.. note::
   In order to reduce processing time, LEOSatpy uses already existing master calibration files if they are available in the ``master_calibs`` subdirectory.
   The user can force to ignore existing master calibration and recreate all master calibration files by using the input option ``-f`` when running ``reduceSatObs``.


WCS calibration in a nutshell
-----------------------------

To perform the astrometric calibration of the reduced FITS-files, use the following command::

    calibrateSatObs PATH/TO/TELESCOPE/DATA/NIGHT_XY

The `calibrateSatObs` command supports the following input options:

* ``-hdu_idx``: Specify the HDU index of the image data in the FITS file. Default is `0`.
* ``-r`` or ``--radius``: Set the download radius for catalog objects in arcminutes. Default is ``auto``.
* ``-c`` or ``--catalog``: Specify the catalog for position reference. Default is `GAIAdr3 <https://www.cosmos.esa.int/web/gaia/data-release-3>`_.
* ``-f`` or ``--force-detection``: Force source catalog extraction.
* ``-d`` or ``--force-download``: Force reference catalog download.
* ``-source_cat_fname``: Name of the source catalog file.
* ``-source_ref_fname``: Name of the reference source catalog file.
* ``-p`` or ``--plot_images``: Show plots during processing.

During the astrometric calibration, LEOSatpy first performs a search for reduced FITS-files in the input path, registers valid image files, and collects instrument settings from the FITS-header keywords.
Based on this information, LEOSatpy then creates an initial guess for the WCS using known telescope characteristics, such as the typical pixel scale of the detector.

Next, LEOSatpy attempts to build a source catalog used in the WCS solution determination. A 2D background map is created using photutils 2D background estimation algorithms, which is then subtracted from the reduced FITS-image.
An initial source detection is performed to identify a set of potential sources on the background-subtracted image.
The :ref:`threshold_value <Source detection options>` for the source detection can be adjusted in the configuration file.
The profiles of these sources are fitted to obtain a revised estimate for the Full Width at Half Maximum (FWHM). Each source is fitted with either a Gaussian or Moffat profile using `LMFIT <https://lmfit.github.io/lmfit-py/>`_ (selected via :ref:`use_gauss <Source detection options>` in the configuration file).
The source detection process is then repeated to obtain a raw source catalog. The profiles of the detected sources are fitted again to obtain centroid positions, FWHM, and signal-to-noise ratios. Sources close to the image borders and each other are removed, and sigma clipping is applied to remove outliers in eccentricity and FWHM.
The remaining sources are then used to estimate the average FWHM of the sources in the image, which is stored and used for further processing, including the trail detection and analysis (see :ref:`Satellite trail analysis in a nutshell`).
Astrometric reference stars with precise positions are obtained from the `GAIA DR3 <https://www.cosmos.esa.int/web/gaia/data-release-3>`_ catalog using `Astroquery <https://astroquery.readthedocs.io/en/latest/>`_.

LEOSatpy uses an iterative approach, adopted from the `Astrometry <https://github.com/lukaswenzl/astrometry>`_ package, to determine the World Coordinate System (WCS) transformation that best matches the detected sources in the image to the reference catalog sources.
The algorithm identifies potential solutions by employing two-dimensional `Fourier–Mellin transform <https://sthoduka.github.io/imreg_fmt/docs/fourier-mellin-transform/>`_ (log-polar FFT) to find the scale and rotation angle, and FFT cross-correlation to estimate the translation shifts.

First, the initial WCS guess is applied to the reference catalog to transform the celestial coordinates into pixel coordinates.
The logarithm of the distances (log(`d`)) and angles (θ) between each source and every other source are calculated for each dataset. These values are then combined using a 2D histogram, forming an image representation of the dataset's.
The Fourier transform of each image is computed, and the cross-power spectrum is formed by multiplying the Fourier transform of source image by the complex conjugate of the reference image.
The inverse Fourier transform of the cross-power spectrum yields an image containing Dirac delta-like peaks. The peak with the highest signal is identified, and the scale and rotation are determined from its location.
The WCS guess is then updated and the translation shifts are determined by performing a phase correlation in the spatial domain.
This process is repeated for multiple rotation patterns, and possible solutions are collected.
These solutions are then analyzed individually, the transformation refined using LMFIT least-squares optimization, and the best solution selected based on the highest quality metrics, such as number of matched sources, completeness, and positional accuracy (RMS error).

Once a solution is found, LEOSatpy updates the FITS header with the new WCS information, saves the calibrated FITS files in the ``calibrated/`` subdirectory, and updates the result table with the calibration results.

.. important::

   When finished, check the FITS-files and figures that LEOSatpy writes to the subdirectories ``calibrated`` and ``figures``.
   The figures show the positions of detected sources and matched reference sources to visually verify the quality of the astrometric solution.
   Sources with a positional accuracy of less than 1 FWHM are marked with green circles, the rest with red circles.

Satellite trail analysis in a nutshell
--------------------------------------

To perform the satellite trail detection and analysis on the calibrated FITS-files, use the following command::

    analyseSatObs PATH/TO/TELESCOPE/DATA/NIGHT_XY

The `analyseSatObs` command supports the following input options:

* ``-hdu_idx``: Specify the HDU index of the image data in the FITS file. Default is `0`.
* ``-b`` or ``--band``: Specify the photometric band to use.
* ``-c`` or ``--catalog``: Specify the catalog for photometric reference. Default is ``auto``. LEOSatpy uses `GSC 2.4.3 <https://ui.adsabs.harvard.edu/#abs/2008AJ....136..735L>`_ by default.
* ``-f`` or ``--force-detection``: Force source catalog extraction.
* ``-d`` or ``--force-download``: Force reference catalog download.
* ``-m`` or ``--manual-select``: Enable manual selection of faint trails.
* ``-photo_ref_fname``: Name of the photometric reference star catalog file.
* ``-p`` or ``--plot_images``: Show plots during processing.

During the analysis, LEOSatpy first searches the input path for calibrated FITS-files, registers them, and collects instrument settings, telescope, and observation information from the FITS-header keywords.
Valid images are grouped by telescope pointing, and each group is processed separately.

LEOSatpy then attempts to automatically detect satellite trails in the calibrated FITS-images.
Unsharp masking (see e.g., `Unsharp masking <https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_unsharp_mask.html>`_) is applied to enhance edges in the background-subtracted FITS image, which helps to identify faint satellite trails.
The unsharp masked image is then segmented, filtered and grouped to identify potential satellite trails.
Each group is then converted into a binary image, and a vectorized Hough transform is applied to map the image pixel coordinates into the Hough parameter space of angles and distances.
Peaks in the Hough space are identified, indicating potential trails, and a sub-region around each peak is extracted and analysed to determine the trail parameters, employing the `Xu et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015PatRe..48.4012X/abstract>`_ algorithm.
Their algorithm uses a statistical approach to analyse the voting distribution in the Hough space, which allows to estimate the trail center position using a linear model, while a quadratic model is used to estimate trail parameters such as length, width, and orientation.
If no trails are detected, the FITS-image is flagged as a reference image that is used to improve the trail photometry.

The brightness of the satellite in the observed filter band is evaluated using aperture photometry, including aperture correction, adopting the `AutoPhOT <https://github.com/Astro-Sean/autophot>`_ algorithm by `Brennan & Fraser (2022) <https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B/abstract>`_.
For this, LEOSatpy compiles a list of sources from the `GSC 2.4.3 <https://ui.adsabs.harvard.edu/#abs/2008AJ....136..735L>`_ catalog that are located within the field of view of the observation.
This photometric reference star catalog is cross-matched with the sources detected in the previous step, and filtered to only include suitable, i.e., bright, isolated, and well-defined sources, with photometric measurements in the observed filter band.

.. important::
   If not enough suitable reference stars are found in the observed filter, LEOSatpy will attempt to estimate the source brightness using color transformation equations by `Lupton et al. (2005) <https://classic.sdss.org/dr4/algorithms/sdssUBVRITransform.php#Lupton2005>`_ and
   `Jester et al. (2005) <https://classic.sdss.org/dr4/algorithms/sdssUBVRITransform.php#Jester2005>`_ to convert the SDSS ugriz magnitudes to UBVRI magnitudes.
   In this case, the magnitude conversion flag ``mag_conv`` in the result table will be set to ``True``.

LEOSatpy then performs a series of measurements on the reference stars, increasing the aperture size step by step, measuring the flux and calculating the signal-to-noise ratio (SNR) for each aperture.
An aperture slightly larger than the one that maximizes the SNR is selected for the flux measurements, to ensure that most of the stellar flux (> 95%) is captured.
Afterwards, the corresponding aperture correction factor is estimated to account for the loss of flux due to the finite size of the aperture.

If **NO** reference image is available, the sources along the trail are masked using the positions of sources from the photometric reference star catalog.
Otherwise, the reference image is used to mask sources along the trail, to minimize contamination of the satellite trail measurements by underlying sources.
In this case, the reference image is first aligned to the trail image, and sources along the trail are masked using the positions of these sources.
The satellite trail flux is then measured using a rectangular aperture that is optimized to capture most of the satellite's flux, similar to the method used for the reference stars.
The results are combined to estimate the observed apparent magnitude of the satellite (``ObsMag``) in the used filter band.

If the Two-Line Element (TLE) file is provided with the data, LEOSatpy will extract the orbital elements of the observed satellite from the TLE file.
These elements are used to calculate the satellite's position, distance, angular velocity, and observation geometry in user-defined steps (see :ref:`dt_step_size <Satellite detection & analysis options>` in the configuration file).
The results of the TLE prediction are stored in the ``tle_predictions`` subdirectory, and the result table is updated with the TLE prediction results taken at the mid-point of the exposure, including:

* ``ObsSatRange``: the estimated distance of the satellite from the observer location in km
* ``SatAngularSpeed``: the estimated angular velocity of the satellite in arcseconds per second
* ``SunSatAng``: the angle between the satellite and the sun in degrees
* ``SunPhaseAng``: the solar phase angle, i.e., the angle between the sun, satellite, and observer in degrees
* ``SunIncAng``: solar incidence angle for a given location on Earth in degrees
* ``ObsAng``: the angle between the satellite and the observer in degrees

LEOSatpy will then use these information to furthermore compute corrections to the observed apparent magnitude (``ObsMag``), such as:

* ``EstMag``: apparent magnitude of the satellite, when corrected for the time the satellite actually spent on the detector, i.e., the time on detector
* ``EstScaleMag``: estimated magnitude of the satellite scaled to a reference altitude (e.g., 550 km for Starlink satellites)

When finished, a final figure is created, showing the detected satellite trail, the used reference stars, and the path predicted by the TLE orbital elements.

.. attention::
   LEOSatpy attempts to fit the satellite trail profile along the trail. However, this is an experimental feature and may not work for all satellite trails.

.. important::
   When finished, check the figures that LEOSatpy writes to the subdirectory ``figures``, as well as the result CSV table in the :ref:`working directory <General options>` defined by :ref:`result_table_name <General options>` in the configuration file.

Congratulations, you have successfully detected and analysed a satellite trail with LEOSatpy!

----

After processing, the directory structure should look like this:
::

    .
    └── Telescope-Identifier
        ├── YYYY-MM-DD
        │   ├── bias
        │   ├── flats
        │   ├── darks
        │   ├── master_calibs
        │   └── science_data (e.g., STARLINK)
        │       ├── auxiliary
        │       ├── calibrated
        │       ├── catalogs
        │       ├── figures
        │       │   └── Sat-ID (e.g., STARLINK-3568)
        │       ├── raw
        │       ├── reduced
        │       └── tle_predictions
        ├── YYYY-MM-DD
        └── YYYY-MM-DD

For a detailed understanding of the configuration options, see the :ref:`Configuration file` section.
