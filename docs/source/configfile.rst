.. _ctio_4m: https://noirlab.edu/science/programs/ctio/telescopes/victor-blanco-4m-telescope

.. |ctio_4m| replace:: Víctor M. Blanco 4-meter Telescope

.. _dk154: https://www.eso.org/public/teles-instr/lasilla/danish154/

.. |dk154| replace:: 1.54-metre Danish telescope

Configuration file
==================

The configuration file (:file:`leosatpy_config.ini`) with the available parameters for LEOSatpy is shown below:

.. include:: ../../leosatpy/leosatpy_config.ini
   :literal:

The available parameters are divided into six groups:

- :ref:`General options`
- :ref:`Reduction options`
- :ref:`Source detection options`
- :ref:`WCS calibration options`
- :ref:`Satellite detection & analysis options`
- :ref:`Figure options`

In the following, we provide a detailed explanation for each parameter present in the configuration file,
outlining its function and valid input options.

.. important::
   LEOSatpy is intended to be highly automated and to work for a wide range of data of varying quality.
   Therefore, aside from the result table name and location, modifying the parameters is generally not required.

General options
---------------

.. _working_dir_path:

.. py:function:: working_dir_path:

   ``str`` (default = ``/home/username/``)

   Specifies the path to the working directory used by LEOSatpy. The primary results table (see result_table_name_) will be saved here.
   `working_dir_path` accepts both absolute and relative paths and defaults to the current user's home directory.

.. important::

   All configuration CSV files—such as regions of interest (roi_table_name_), extensions of interest (ext_oi_table_name_), glint masks (glint_mask_table_name_), and faint trail lists (faint_trail_table_name_)—must be placed in this directory.

.. _result_table_name:

.. py:function:: result_table_name:

   ``str`` (default = ``results_leosatpy.csv``)

   Name of the CSV file that stores the results of the analysis. This file is the main result of LEOSatpy and contains all information about the processed images, including metadata, measured quantities, and derived results.
   The file will be created in the directory specified by working_dir_path_.

.. _roi_table_name:

.. py:function:: roi_table_name:

   ``str`` (default = ``leosatpy_regions_of_interest.csv``)

   Name of the CSV file listing specific image regions (by pixel coordinates) to be analyzed. If this file is present in the working_dir_path_, LEOSatpy will only process the specified regions of interest when the list of images is processed.
   The user can manually create this file to focus the analysis on specific areas of interest within larger images, helping to reduce processing time and memory usage.
   To define regions of interest, the CSV file should contain the following columns:

   - *File*: Name of the image file.
   - *xstart*, *xend*: Start and end pixel coordinates in the x-direction.
   - *ystart*, *yend*: Start and end pixel coordinates in the y-direction.


   .. list-table:: Example for a region of interest selection.
       :header-rows: 1

       * - File
         - xstart
         - xend
         - ystart
         - yend
       * - 20221112T125843C1o
         - 0
         - 800
         - 4750
         - 6143

   .. attention::
      LEOSatpy uses zero-based indexing for pixel coordinates, meaning the first pixel in an image is at (0, 0).

   .. warning::
      Duplicate file name entries and files with multiple HDU's are currently not supported.

.. _ext_oi_table_name:

.. py:function:: ext_oi_table_name:

   ``str`` (default = ``leosatpy_extensions_of_interest.csv``)

   Name of the CSV file specifying which extensions (HDU's) in multi-extension FITS files should be analyzed. If this file is present in the working_dir_path_, LEOSatpy will only process the specified extensions in the listed images.
   This is in particular useful for multi extension FITS files, e.g., from the |ctio_4m|_, where multiple HDUs may contain satellite trails or other relevant data. This file allows users to specify which HDUs should be considered for analysis, thus avoiding unnecessary processing of irrelevant data.
   To define the extensions of interest, the CSV file should contain the following columns:

   - *File*: Filename of the image with satellite trails
   - *HDUs*: Python list format string containing the indices of HDUs with satellite trails

   .. list-table:: Example for extensions of interest
       :header-rows: 1

       * - File
         - HDUs
       * - 38273f0c3320d3d836930c231ed930fe_c4d_231201_074349_opi_i_v1
         - "[7,12,13,48,49]"
       * - 3eaf91433036cabf9bf5971f98fe2643_c4d_231201_081149_opi_g_v1
         - "[6,10,11]"

   .. attention::
      The HDU indices are zero-based, meaning the first HDU is indexed as 0, the second as 1, and so on.

.. _glint_mask_table_name:

.. py:function:: glint_mask_table_name:

   ``str`` (default = ``leosatpy_mask_glints.csv``)

   Name of the CSV file specifying parts of the trail that show bright reflections (glints) and should be masked and ignored during the photometric analysis.
   If present, each row defines a rectangular aperture based on photutils `RectangularAperture <https://photutils.readthedocs.io/en/latest/api/photutils.aperture.RectangularAperture.html#photutils.aperture.RectangularAperture>`_, required to mask the glint in the image.
   To define a glint mask, the CSV file should contain the following columns:

   - *File*: Name of the image file.
   - *HDU*: HDU index containing the glint.
   - *xc*, *yc*: Center coordinates of the mask (pixels).
   - *length*, *width*: Size of the mask (pixels).
   - *ang*: Rotation angle of the mask (degrees from the x-axis).

   .. list-table:: Example for masking glints in a satellite trail.
       :header-rows: 1

       * - File
         - HDU
         - xc
         - yc
         - length
         - width
         - ang
       * - Starlink-30144_000002
         - 0
         - 346.67
         - 1013.33
         - 600
         - 75
         - 155.25
       * - Oneweb-0108_000001
         - 0
         - 1492
         - 423
         - 900
         - 60
         - 64.5

   .. attention::
      LEOSatpy uses zero-based indexing. This means that the first HDU in a FITS file is indexed as 0, the second as 1, and so on, while pixel coordinates start at (0, 0).


.. _faint_trail_table_name:

.. py:function:: faint_trail_table_name:

   ``str`` (default = ``leosatpy_faint_trails.csv``)

   Name of the CSV file listing images containing faint satellite trails. If this file is present in the working_dir_path_,
   each entry defines a rectangular aperture used to mark a faint trail in the image, that is not detected automatically by the algorithm.
   To define a faint trail aperture, the CSV file should contain the following columns:

   - *File*: Name of the image file.
   - *HDU*: Index of the HDU extension containing the faint trail.
   - *xc*, *yc*: Center coordinates of the trail (pixels).
   - *length*, *width*: Size of the trail area (pixels).
   - *ang*: Rotation angle of the trail (degrees from the x-axis).

   .. list-table:: Example for faint trails
       :header-rows: 1

       * - File
         - HDU
         - xc
         - yc
         - length
         - width
         - ang
       * - Starlink-5464_000001
         - 0
         - 1024
         - 1048
         - 700
         - 10
         - 135.2
       * - Oneweb-0350_000001
         - 0
         - 1100
         - 850
         - 600
         - 20
         - 22.5

   .. attention::
      LEOSatpy uses zero-based indexing. This means that the first HDU in a FITS file is indexed as 0, the second as 1, and so on, while pixel coordinates start at (0, 0).

.. py:function:: def_select_timeout:

   ``int`` (default = ``5``)

   Default timeout (in seconds) for selection dialogs in the user interface. Set to ``-1`` to disable the timeout and wait indefinitely for user input.

Reduction options
-----------------

.. py:function:: timedelta_days:

   ``int`` (default = ``7``)

   Specifies the date range (± days) around the observation date to search for calibration files.

.. py:function:: bias_correct:

   ``True``, ``False`` (default = ``True``)

   If ``True``, applies bias correction to remove the inherent electronic offset in the CCD detector.

.. py:function:: dark_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``, subtracts dark frames to remove thermal noise generated by the detector.

.. py:function:: flat_correct:

   ``True``, ``False`` (default = ``True``)

   If ``True``, applies flat field correction to account for pixel-to-pixel sensitivity variations and optical vignetting.

.. py:function:: flatdark_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``, performs dark subtraction on corresponding flat field frames before using them for calibration. This correction removes thermal noise contribution from flat fields.

.. important::

   If one of the parameter above is ``True`` and no `bias`, `dark`, or `flat` FITS-files are found,
   the corresponding parameter is automatically set to ``False``.

.. _correct_gain:

.. py:function:: correct_gain:

   ``True``, ``False`` (default = ``False``)

   If ``True`` and the gain is known, applies gain correction to convert pixel values from ADU (Analog-to-Digital Units) to electrons.

.. _correct_cosmic:
.. py:function:: correct_cosmic:

   ``True``, ``False`` (default = ``False``)

   If ``True``, identifies and removes cosmic ray hits using either the L.A. Cosmic algorithm (based on Laplacian edge detection) or a median filtering technique.

.. _est_uncertainty:
.. py:function:: est_uncertainty:

   ``True``, ``False`` (default = ``False``)

   If ``True``, creates an uncertainty frame that incorporates read noise, Poisson (photon) noise, and other error sources.

.. warning::
   The parameters correct_gain_, correct_cosmic_, and est_uncertainty_ are not yet fully implemented and
   will become available in a future release.

.. py:function:: combine_method_bias:

   ``'average'``, ``'median'``, ``'sum'`` (default = ``'median'``)

   Method to combine multiple `BIAS`-frames into a master bias:

       - ``'average'`` : Combines frames by calculating the arithmetic mean, which can improve signal-to-noise ratio but is sensitive to outliers.
       - ``'median'`` : Combines frames by calculating the median value, effectively rejecting outliers like cosmic rays.
       - ``'sum'`` : Combines frames by calculating the sum, which preserves total counts but typically requires normalization.

.. py:function:: combine_method_dark:

   ``'average'``, ``'median'``, ``'sum'`` (default = ``'average'``)

   Method to combine multiple `DARK`-frames into a master dark:

       - ``'average'`` : Combines frames by calculating the arithmetic mean, optimizing signal-to-noise ratio when frames have consistent thermal noise patterns.
       - ``'median'`` : Combines frames by calculating the median value, rejecting outliers but potentially undercorrecting for thermal noise.
       - ``'sum'`` : Combines frames by calculating the sum, preserving total counts but typically requiring normalization.

   .. :file:`/home/username/`

.. _combine_method_flat:

.. py:function:: combine_method_flat:

   ``'average'``, ``'median'``, ``'sum'`` (default = ``'median'``)

   Method to combine multiple `FLAT`-frames into a master flat:

       - ``'average'`` : Combines frames by calculating the arithmetic mean, optimizing signal-to-noise ratio but potentially including star residuals.
       - ``'median'`` : Combines frames by calculating the median value, effectively removing star residuals and other transient features.
       - ``'sum'`` : Combines frames by calculating the sum, preserving total counts but typically requiring normalization.


   .. :file:`/home/username/`

.. py:function:: vignette_frac:

   ``float`` (default = ``0.975``)

   This parameter defines the fraction of the image area included in the vignetting mask. This mask is only applied to U-band observations from the |dk154|_ to exclude the outer regions of the image where vignetting effects are most pronounced in this band.
   The default value of 0.975 means that 97.5% of the image area is included in the mask, effectively excluding the outer 2.5% of the image from the analysis.

.. py:function:: mem_lim_combine:

   ``float`` (default = ``8e9``)

   The maximum memory (in bytes) that should be allocated when combining frames using the
   `Ccdproc <https://ccdproc.readthedocs.io/en/stable/>`_ combine method. Adjust this value based on your system's available RAM to prevent memory errors when processing large image stacks.


Source detection options
------------------------

.. py:function:: bkg_box_size:

   ``int`` (default = ``25``)

   The box size (in pixels) used to estimate the background level via sigma-clipped statistics. This parameter controls the resolution of the background map, with larger values producing smoother maps but potentially missing small-scale background variations.


.. py:function:: bkg_med_win_size:

   ``int`` (default = ``5``)

   The window size of the 2D median filter applied to the low-resolution background map. This filter smooths the background estimation and reduces the impact of small sources. Passed as the
   ``filter_size`` parameter in `Background2D <https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html>`_.

.. py:function:: source_box_size:

   ``int`` (default = ``21``)

   Initial size (in pixels) of the cutout image used for fitting the FWHM of detected sources. This value is adaptively updated during processing based on the measured FWHM to ensure appropriate fitting regions for stars.


.. py:function:: source_min_no:

   ``int`` (default = ``5``)

   The minimum number of sources required when performing source detection for FWHM determination. If fewer sources are found, the detection threshold is automatically adjusted to identify more sources.

.. py:function:: source_max_no:

   ``int`` (default = ``1000``)

   The maximum number of sources allowed when performing source detection for FWHM determination. This limit helps prevent excessive computational load and ensures that only the brightest, most reliable sources are used for aperture photometry.


.. py:function:: fwhm_init_guess:

   ``float`` (default = ``5.``)

   Initial estimate of the source FWHM (in pixels) used by the detection algorithm. Once sources are detected, this value is refined based on actual measurements. An appropriate initial guess improves the efficiency of source detection.


.. py:function:: fwhm_lim_min:

   ``float`` (default = ``1.``)

   Lower limit (in pixels) for FWHM fitting. Sources with fitted FWHM below this value are rejected as they likely represent noise spikes, hot pixels, or cosmic rays rather than actual astronomical sources.


.. py:function:: fwhm_lim_max:

   ``float`` (default = ``30.``)

   Upper limit (in pixels) for FWHM fitting. Sources with fitted FWHM above this value are rejected as they likely represent extended objects, source blends, or fitting failures rather than point sources.

.. _threshold_value:
.. py:function:: threshold_value:

   ``float`` (default = ``3.``)

   Initial detection threshold, expressed as a multiple of the background standard deviation (sigma). Sources with peak intensities below this threshold are ignored. This value is dynamically adjusted during processing to achieve an optimal number of detected sources.


.. py:function:: isolate_sources_init_sep:

   ``float`` (default = ``20.``)

   Minimum separation (in pixels) between detected sources during initial source finding. Sources with neighbors within this distance are rejected to ensure clean, isolated sources for accurate aperture photometry.


.. py:function:: isolate_sources_fwhm_sep:

   ``float`` (default = ``5.``)

   Minimum separation between sources expressed as a multiple of the measured FWHM. This adaptive criterion ensures that sources are sufficiently isolated for aperture photometry.

.. _use_gauss:
.. py:function:: use_gauss:

   ``True``, ``False`` (default = ``True``)

   If ``True``, uses Gaussian fitting for source detection and characterization. If ``False``, uses Moffat profiles instead. Gaussian profiles are computationally faster but Moffat profiles often provide better fits to astronomical point spread functions.


.. py:function:: fitting_method:

   ``str`` (default = ``'least_square'``)

   The numerical method used for analytical function fitting via `LMFIT <https://lmfit.github.io/lmfit-py/fitting.html>`_.
   Options include:

   - ``'leastsq'`` : Levenberg-Marquardt algorithm, good for general purpose fitting
   - ``'least_squares'`` : Trust Region Reflective method, handles bounds effectively
   - ``'powell'`` : Powell's method, derivative-free optimization suitable for noisy data
   - ``'nelder'`` : Nelder-Mead simplex algorithm, robust for non-smooth functions


.. py:function:: default_moff_beta:

   ``float`` (default = ``4.765``)

   The beta parameter for Moffat profiles, controlling the behavior of the profile wings. The default value of ``4.765`` is based on `Trujillo et al. (2001) <https://doi.org/10.1046/j.1365-8711.2001.04937.x>`_ and is appropriate for most astronomical seeing conditions. Lower values produce heavier wings (a value of 1 corresponds to a Lorentzian profile).

WCS calibration options
-----------------------

.. py:function:: ref_catalog_mag_lim:

   ``-1``, ``int`` (default = ``19``)

   Magnitude limit for reference catalog sources. Sources fainter than this limit are excluded from the astrometric solution. Setting to ``-1`` disables magnitude filtering and uses all available catalog sources.

.. note::
   While limiting the reference catalog to brighter sources can significantly reduce download times, choosing the limit too low may result in fewer matched sources, especially in sparse fields.

.. py:function:: distance_bin_size:

   ``float`` (default = ``1.``)

   Bin size (in pixels) for the distance histogram used in the pattern-matching algorithm for astrometric calibration. This parameter affects the resolution of the distance correlation between observed and reference star patterns. Smaller values provide higher precision but require more computation.


.. py:function:: ang_bin_size:

   ``float`` (default = ``0.1``)

   Bin size (in degrees) for the angle histogram used in the pattern-matching algorithm. This parameter controls the angular resolution when comparing star pattern geometries between observed and reference catalogs. Smaller values provide more precise pattern matching but increase computational requirements.

.. _match_radius:
.. py:function:: match_radius:

   ``fwhm``, ``float`` (default = ``'fwhm'``)

   The maximum distance (in pixels) within which an observed source and reference catalog source are considered a match. Using ``'fwhm'`` dynamically sets this value to the measured seeing FWHM, which adapts the matching tolerance to the image quality. Alternatively, a fixed pixel value can be specified.

.. py:function:: match_radius_lim:

   ``float`` (default = ``3.``)

   Maximum match radius when updating the catalogs, expressed as a multiple of the match_radius_. This parameter sets an upper limit on how far apart two sources can be and still be considered a match during catalog updates.

.. _initial_source_lim:
.. py:function:: initial_source_lim:

   ``int`` (default = ``100``)

   Initial limit on the number of sources to consider when attempting to find a valid astrometric solution. This parameter helps control computational load by limiting the number of sources processed in the initial stages of the WCS calibration.

.. py:function:: max_ref_num:

   ``int`` (default = ``500``)

   Maximum number of reference catalog sources to use in the astrometric solution. Limiting this number improves computational efficiency while maintaining solution accuracy, especially in dense star fields.

.. py:function:: num_ref_samples:

   ``int`` (default = ``5``)

   Number of reference samples to be drawn from the available data when attempting to find a valid astrometric solution.

.. py:function:: max_no_solution_count:

   ``int`` (default = ``3``)

   Maximum number of attempts to find a valid astrometric solution before considering alternative parameters or approaches.

.. py:function:: max_num_top_solution:

   ``int`` (default = ``3``)

   Maximum number of top-ranked potential astrometric solutions to evaluate before selecting the best one. Higher values provide more thorough consideration of possible solutions but increase computation time.

.. py:function:: min_source_no_convergence:

   ``int`` (default = ``4``)

   Minimum number of matched sources required to consider the WCS solution valid. At least this many sources must be successfully matched between the image and reference catalog before accepting a solution. A higher value enforces more robust solutions but may fail in sparse fields.

.. py:function:: threshold_convergence:

   ``float`` (default = ``0.95``)

   Convergence threshold for the astrometric solution, expressed as the fraction of detected sources that must be successfully matched to reference catalog sources. The solution is considered successful when this percentage (95% by default) of sources is matched.


Satellite detection & analysis options
---------------------------------------

.. py:function:: sharpen_amount:

   ``int`` (default = ``15``)

   Intensity factor for the unsharp masking filter used to enhance satellite trail detection. This technique identifies fine details by computing the difference between the original image and a blurred version, then adds these enhanced details back to the original. Higher values amplify faint trails but may also increase noise. A value of 0 disables sharpening, while negative values can be used for smoothing effects.

.. _n_trails_max:
.. py:function:: n_trails_max:

   ``int`` (default = ``1``)

   Maximum number of satellite trails to detect and analyze in a single image. When multiple trails are present, the algorithm will identify up to this many trails, prioritizing by detection confidence. For images expected to contain multiple satellite trails (e.g., from satellite constellations), increase this value.

.. attention::
   Currently only the first detected trail is analyzed, even if multiple trails are found. The `parallel_trail_select`_ parameter can be used to select which trail to analyze when multiple parallel trails are detected.

.. _parallel_trail_select:

.. py:function:: parallel_trail_select:

   ``int`` (default = ``0``)

   Index of the trail to select when multiple parallel trails are detected. In cases where several trails have similar angles and positions (e.g., from a satellite train), this index specifies which one to analyze, where 0 represents the first detected trail.

.. _max_distance:
.. py:function:: max_distance:

   ``int`` (default = ``2``)

   Maximum angular deviation (in degrees) allowed when grouping detected trail segments into a single trail. Trail segments with angular differences less than ±(max_distance/2) degrees will be considered part of the same trail. Increase for curved trails or trails with gaps; decrease for more stringent grouping in crowded fields.

.. _rho_bin_size:
.. py:function:: rho_bin_size:

   ``float`` (default = ``0.5``)

   Size of the distance bin (in pixels) used in the Hough Transform algorithm for linear feature detection. This parameter affects the resolution of the distance parameter in the transform space. Smaller values provide finer detail in detecting trails but require more computation and memory.

.. _theta_bin_size:
.. py:function:: theta_bin_size:

   ``float`` (default = ``0.025``)

   Size of the angle bin (in radians) used in the Hough Transform algorithm. Controls the angular resolution for detecting linear features. Smaller values allow for more precise angle determination of satellite trails but increase computational requirements.

.. _rho_sub_win_res_fwhm:
.. py:function:: rho_sub_win_res_fwhm:

   ``float`` (default = ``6.5``)

   Resolution of the distance parameter (expressed as a multiple of the FWHM) for the Hough Transform sub-window. This parameter controls the refinement step after initial detection to precisely localize the trail. Larger values provide better trail localization but require more memory.

.. _theta_sub_win_size:
.. py:function:: theta_sub_win_size:

   ``float`` (default = ``3.``)

   Size of the angle sub-window (in degrees) for the Hough Transform refinement step. After initial trail detection, this parameter defines the angular range to search for the precise trail orientation. Larger values accommodate more uncertainty in the initial detection.

.. py:function:: trail_params_fitting_method:

   ``str`` (default = ``'least_square'``)

   Numerical method used for fitting the satellite trail parameters. Similar to the source fitting method, options include:

   - ``'leastsq'`` : Levenberg-Marquardt algorithm, generally robust for trail profile fitting
   - ``'least_squares'`` : Trust Region Reflective method, handles constraints well
   - ``'powell'`` : Powell's method, useful for noisy trail data
   - ``'nelder'`` : Nelder-Mead simplex algorithm, more resistant to local minima

.. py:function:: num_std_min:

   ``int`` (default = ``5``)

   Minimum number of comparison stars required for photometric calibration of satellite trails. Fewer than this many suitable reference stars will result in a warning about potentially unreliable photometry. Increase in sparse fields if calibration is failing.

.. py:function:: num_std_max:

   ``int`` (default = ``500``)

   Maximum number of comparison stars to use for photometric calibration. Using too many stars can slow down processing without significantly improving accuracy. In dense fields, limiting this number improves efficiency while maintaining photometric precision.

.. py:function:: aper_rad:

   ``float`` (default = ``1.7``)

   Default aperture radius for photometry, expressed as a multiple of the image FWHM. This adaptive approach ensures appropriate aperture sizes regardless of seeing conditions. The default value of 1.7×FWHM typically captures ~95% of a stellar flux, balancing completeness with background noise minimization.

.. py:function:: inf_aper_rad:

   ``float`` (default = ``2.5``)

   Radius for the "infinite" aperture used for aperture correction, expressed as a multiple of the FWHM. Although not truly infinite, this larger aperture captures nearly all of the source flux for accurate corrections to the standard aperture measurements. Must be larger than ``aper_rad`` to provide meaningful corrections.

.. py:function:: aper_start:

   ``float`` (default = ``0.1``)

   Starting radius (as a multiple of FWHM) for the optimal aperture determination process. The algorithm tests apertures beginning with this size to find the optimal signal-to-noise ratio for photometry.

.. py:function:: aper_stop:

   ``float`` (default = ``5``)

   Maximum radius (as a multiple of FWHM) to consider when determining the optimal aperture size. The algorithm will not test apertures larger than this value, as they typically include too much sky background.

.. py:function:: aper_step_size:

   ``float`` (default = ``0.1``)

   Step size (as a fraction of FWHM) between consecutive aperture radii when determining the optimal aperture. Smaller values provide finer granularity but increase computation time. The algorithm tests apertures from ``aper_start`` to ``aper_stop`` in increments of this size.

.. _rskyin:
.. py:function:: rskyin:

   ``float`` (default = ``2.``)

   Inner radius of the annulus used for background estimation in aperture photometry, expressed as a multiple of the FWHM. This radius should be larger than the aperture radius to avoid including source flux in the background estimation.

.. _rskyout:
.. py:function:: rskyout:

   ``float`` (default = ``3.``)

   Outer radius of the annulus used for background estimation, expressed as a multiple of the FWHM. The algorithm estimates the local background from pixels between ``rskyin`` and ``rskyout``. Should be larger than ``rskyin`` to provide adequate sampling of the background.

.. _dt_step_size:
.. py:function:: dt_step_size:

   ``float`` (default = ``0.01``)

   Time step size (in seconds) used for calculating the angular velocity of satellites.

.. py:function:: trail_aper_rad:

   ``float`` (default = ``1.5``)

   Default aperture radius for trail photometry, expressed as a multiple of the image FWHM. This adaptive approach ensures appropriate aperture sizes for satellite trails regardless of seeing conditions.

.. py:function:: inf_trail_aper_rad:

   ``float`` (default = ``5``)

   Radius for the "infinite" aperture used for trail aperture correction, expressed as a multiple of the FWHM. This larger aperture captures nearly all of the trail flux for accurate corrections to the standard aperture measurements.

.. py:function:: trail_rskyin:

   ``float`` (default = ``5.``)

   Inner radius of the annulus used for background estimation in trail photometry, expressed as a multiple of the FWHM. This radius should be larger than the trail aperture radius to avoid including trail flux in the background estimation.

.. py:function:: trail_rskyout:

   ``float`` (default = ``6.``)

   Outer radius of the annulus used for background estimation in trail photometry, expressed as a multiple of the FWHM. The algorithm estimates the local background from pixels between ``trail_rskyin`` and ``trail_rskyout``.

.. py:function:: trail_aper_start:

   ``float`` (default = ``0.1``)

   Starting radius (as a multiple of FWHM) for the optimal trail aperture determination process. The algorithm tests apertures beginning with this size to find the optimal signal-to-noise ratio for trail photometry.

.. py:function:: trail_aper_stop:

   ``float`` (default = ``10.``)

   Maximum radius (as a multiple of FWHM) to consider when determining the optimal trail aperture size. The algorithm will not test apertures larger than this value.

.. py:function:: trail_aper_step_size:

   ``float`` (default = ``0.1``)

   Step size (as a fraction of FWHM) between consecutive aperture radii when determining the optimal trail aperture. Smaller values provide finer granularity but increase computation time.

Figure options
---------------

.. py:function:: fig_type:

   ``'png'``, ``'pdf'`` (default = ``'png'``)

   File format for saving output figures. PNG format is preferred for web display and general purpose use, while PDF is vector-based and better suited for publication-quality figures and documents where scaling without loss of quality is required.

.. py:function:: fig_size:

   `tuple` (default = ``(10, 6)``)

   .. ``'automatic'``, `tuple` (default = ``'automatic'``)

   Tuple of 2 parameters pertaining to the size of the figures

.. py:function:: fig_dpi:

   ``int`` (default = ``150``)

   Resolution of output figures in dots per inch (DPI).

.. py:function:: line_length:

   ``float`` (default = ``5.``)

   Length of image scale indicator.

.. py:function:: arrow_length:

   ``float`` (default = ``0.15``)

   Length of the compass arrows in the image plots, specified as a fraction of the figure size. These arrows indicate the North and East directions on the sky.

.. py:function:: manual_trail_select_zoom_lvl:

``float`` (default = ``0.1``)

   Zoom level for the manual trail selection figure. This value is a fraction of the original image size.
