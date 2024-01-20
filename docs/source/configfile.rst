
Configuration file
==================


The configuration file (:file:`leosatpy_config.ini`) with the available parameters for LEOSatpy is shown below:

.. include:: ../../leosatpy/leosatpy_config.ini
   :literal:

The available parameter are divided into six groups:

- :ref:`General options`
- :ref:`Reduction options`
- :ref:`Source detection options`
- :ref:`WCS calibration options`
- :ref:`Satellite detection & analysis options`
- :ref:`Figure options`

In the following, we provide a detailed explanation for each parameter present in the configuration file,
outlining its function and valid input options.

.. important::

    LEOSatpy is highly automated.
    Therefore, aside from the result table name and location, modifying the parameter is generally not required.

.. *H*\ :sub:`orb`,
.. `ccdproc <https://ccdproc.readthedocs.io/en/latest/>`_


General options
---------------

.. _result_table_path:

.. py:function:: result_table_path:

    ``str`` (default = ``/home/username/``)

    :file:`/home/username/` The directory to be used for storing the result.csv table.

.. _result_table_name:

.. py:function:: result_table_name:

    ``str`` (default = ``results_leosatpy.csv``)

    :file:`/home/username/` Name of the result .csv table. with all relevant information on the observation analysed object and
    results of aperture photometry of the sat trail if detected

.. _roi_table_name:

.. py:function:: roi_table_name:

    ``str`` (default = ``leosatpy_regions_of_interest.csv``)

    Name of .csv file containing manually selected trails.

    - file_name,
    - xstart,
    - xend,
    - ystart,
    - yend

    .. warning::
        Duplicate file name entries are not supported at the moment.


Reduction options
-----------------

.. py:function:: bias_correct:

    ``True``, ``False`` (default = ``False``)

    If ``True``, apply bias correction.

.. py:function:: dark_correct:

    ``True``, ``False`` (default = ``False``)

    If ``True``, subtract dark

.. py:function:: flat_correct:

    ``True``, ``False`` (default = ``False``)

    If ``True``, apply flat division

.. py:function:: flat_dark_correct:

    ``True``, ``False`` (default = ``False``)

    If ``True``, perform dark subtraction on flat

.. py:function:: overscan_correct:

    ``True``, ``False`` (default = ``False``)

    If ``True``, apply overscan subtraction.
    is ignored if the telescope is known to have the overscan included in zhe bias or dark

.. important::

    If one of the parameter above is ``True`` and **No** `bias`, `dark`, or `flat` FITS-files are found,
    the corresponding parameter is automatically set to ``False``.

.. py:function:: correct_gain:

    ``True``, ``False`` (default = ``False``)

    If ``True`` and the gain is known, correct the gain in the image.

.. py:function:: correct_cosmic:

    ``True``, ``False`` (default = ``False``)

    If ``True``, Identify cosmic rays either through the L.A. Cosmic or the median technique.

.. py:function:: est_uncertainty:

    ``True``, ``False`` (default = ``False``)

    If ``True``, create an uncertainty frame.

.. py:function:: combine_method_bias:

    ``average``, ``median``, ``sum`` (default = ``median``)

    Method to combine BIAS-frames

.. py:function:: combine_method_dark:

    ``average``, ``median``, ``sum`` (default = ``average``)

    .. :file:`/home/username/`

.. _combine_method_flat:

.. py:function:: combine_method_flat:

    ``average``, ``median``, ``sum`` (default = ``median``)

    .. :file:`/home/username/`




Source detection options
------------------------

.. py:function:: bkg_box_size:

    ``int`` (default = ``25``)

    The box size used to estimate the (sigma-clipped) statistics to create a low-resolution background map.


.. py:function:: bkg_med_win_size:

    ``int`` (default = ``5``)

    The window size of the 2D median filter to apply to the low-resolution background map as the
    ``filter_size`` parameter in `Background2D <https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html>`_.
    `photutils <https://photutils.readthedocs.io/en/stable/>`_

.. py:function:: source_box_size:

    ``int`` (default = ``21``)

    Initial image size in pixels to take cutout for fitting the FWHM. This is updated during the automated script.


.. py:function:: source_min_no:

    ``int`` (default = ``5``)

    When performing source detection,
    what is the minimum allowed sources when doing source detection to find fwhm.

.. py:function:: source_max_no:

    ``int`` (default = ``1000``)

    When performing source detection,
    what is the maximum allowed sources when doing source detection to find fwhm.
    This value dictates how the ``threshold_value`` behaves.

Commands to control source detection algorithm used for finding bright, isolated stars. This list of stars is used when building the PSF, finding the FWHM and solving for the WCS.

.. py:function:: fwhm_init_guess:

    ``float`` (default = ``5.``)

    Source detection algorithms need an initial guess for the FWHM. Once any sources are found, we find an approximate value for the FWHM and update our source detection algorithm.


.. py:function:: fwhm_lim_min:

    ``float`` (default = ``1.``)

    lower limit for fwhm fit

.. py:function:: fwhm_lim_max:

    ``float`` (default = ``30.``)

    When fitting for the FWHM, constrain the fitting to allow for this maximum value to fit
    for the FWHM.

.. py:function:: threshold_value:

    ``float`` (default = ``10.``)

    An appropriate threshold value is needed to detection bright sources.
    This value is the initial threshold level for source detection.
    This is just an initial guess and is update incrementally until an useful number of sources
    is found.


.. py:function:: threshold_value_lim:

    ``float`` (default = ``3.``)

    This is the lower limit on the threshold value.
    If the threshold value decreases below this value, use *fine_fudge_factor*.
    This is a safety features if an image contains few stars above the background level.
    For example there may be no sources at *threshold_value=4* but a few are detected at *threshold_value=4.1*.


.. py:function:: threshold_fudge_factor:

    ``float`` (default = ``5.``)

    large step for source detection

.. py:function:: threshold_fine_fudge_factor:

    ``float`` (default = ``0.1``)

    small step for source detection if required

.. py:function:: sigmaclip_fwhm_sigma:

    ``float`` (default = ``3.``)

    When cleaning the FWHM measurements of the found sources in a image, using sigma-clipped statistics to sigma clip the values for the FWHM by this amount.


.. py:function:: isolate_sources_init_sep:

    ``float`` (default = ``20.``)

    For initial guess, sources are removed if they have a detected neighbour within this value, given in pixels.


.. py:function:: isolate_sources_fwhm_sep:

    ``float`` (default = ``5.``)

    When a sample of sources is found, separate sources by this amount times the FWHM.


.. py:function:: max_func_iter:

    ``int`` (default = ``50``)

    Maximum number of iterations the source detection algorithm is performed.


.. py:function:: fitting_method:

    ``str`` (default = ``least_square``)

    analytical function fitting using `LMFIT <https://lmfit.github.io/lmfit-py/fitting.html>`_
    We can accept a limited number of methods from `here <https://lmfit.github.io/lmfit-py/fitting.html>`_.
    Fitting method for analytical function fitting and PSF fitting. We can accept a limited number of methods from `here <https://lmfit.github.io/lmfit-py/fitting.html>`_. Some tested methods including: \n\n\t * leastsq \n\t * least_squares \n\t * powell \n\t * nelder


.. py:function:: default_moff_beta:

    ``float`` (default = ``4.765``)

    `Trujillo et al. (2001) <https://doi.org/10.1046/j.1365-8711.2001.04937.x>`_
    If *use_moffat* is True, set the beta term which describes hwo the *wings* of the moffat function behave. We pre-set this to `4.765 <https://academic.oup.com/mnras/article/328/3/977/1247204>`_. IRAF defaults this value to 2.5. A Lorentzian can be obtained by setting this value to 1.


(e.g. :math:`f>5\sigma`)

WCS calibration options
-----------------------


.. py:function:: ref_catalog_mag_lim:

    ``median`` (default = ``results_leosatpy.csv``)

    :file:`/home/username/`
    Ignore catalog sources that have a given magnitude (i.e. not measured) lower than this value.
    This is used to decrease computation time, by ignoring sources that are expected to be too faint.


.. py:function:: max_wcs_func_iter:

    ``int`` (default = ``10``)

    Max number of attempts to find a solution

.. py:function:: ref_sources_max_no:

    ``int`` (default = ``2500``)

    Maximum number of reference sources to use

.. py:function:: max_no_obs_sources:

    ``-1``, ``int`` (default = ``-1``)

    Maximum number of observed sources to use

.. py:function:: distance_bin_size:

    ``float`` (default = ``1.``)

    bin size of distances for 2d hist

.. py:function:: ang_bin_size:

    ``float`` (default = ``0.1``)

    bin size of angles for 2d hist

.. py:function:: match_radius:

    ``fwhm``, ``float`` (default = ``fwhm``)

    match radius within which source and ref are considered matched

.. py:function:: min_source_no_convergence:

    ``int`` (default = ``5``)

    Minimum number required to trigger convergence

.. py:function:: threshold_convergence:

    ``float`` (default = ``0.95``)

    Minimum percentage of detected sources are matched

Satellite detection & analysis options
---------------------------------------


.. py:function:: n_trails_max:

    ``int`` (default = ``1``)

    Maximum number of trails to extract.  Maximum number of trails expected in the observation

.. py:function:: parallel_trail_select:

    ``int`` (default = ``0``)

    :file:`/home/username/`
    Index of trail to select from multiple parallel detections


.. py:function:: max_distance:

    ``int`` (default = ``2``)

    Maximum angular deviation to be selected into group of segments
    Angle for detection grouping of trail segments, +-Angle = max_distance / 2

.. py:function:: rho_bin_size:

    ``float`` (default = ``0.5``)

    Size of distance bin in HT

.. py:function:: theta_bin_size:

    ``float`` (default = ``0.02``)

    Size of angle bin in HT


.. py:function:: rho_sub_win_res_fwhm:

    ``float`` (default = ``6.``)

    Size of distance bin in HT sub window

.. py:function:: theta_sub_win_size:

    ``float`` (default = ``3.``)

    Size of angle bin in HT sub window

.. py:function:: trail_params_fitting_method:

    ``str`` (default = ``least_square``)

    Fitting method for analytical function fitting

.. py:function:: num_std_min:

    ``int`` (default = ``5``)

    Minimum number of comparison stars

.. py:function:: num_std_max:

    ``int`` (default = ``500``)

    Maximum number of comparison stars to consider

.. py:function:: aper_rad:

    ``float`` (default = ``1.7``)

    Default Aperture size. This is taken as the multiple of the image full width half maximum.


.. py:function:: inf_aper_rad:

    ``float`` (default = ``2.5``)

    Default *infinite* aperture size used for aperture correction. Although this is not infinite in size, it is assumed large enough to capture significantly larger flux than the standard aperture size. Must be larger than *ap_size*. Cannot be larger than *scale_multipler*


.. py:function:: aper_start:

    ``float`` (default = ``0.1``)

    Start aperture radius for optimum aperture estimation

.. py:function:: aper_stop:

    ``float`` (default = ``5``)

    Start aperture radius for optimum aperture estimation

.. py:function:: aper_step_size:

    ``float`` (default = ``0.1``)

    Step size

.. py:function:: rskyin:

    ``float`` (default = ``2.``)

    Inner radius of annulus for background estimate when performing aperture photometry. Should be slightly larger than the aperture size (*ap_size*)

.. py:function:: rskyout:

    ``float`` (default = ``3.``)

     Outer radius of annulus for background estimate when performing aperture photometry. Should be slightly larger than the aperture size (*ap_size*) and r_in_size


.. py:function:: dt_step_size:

    ``float`` (default = ``0.01``)

    Time step size for angular velocity calculation in seconds.

Figure options
---------------

.. py:function:: fig_type:

    ``png``, ``pdf`` (default = ``png``)

    Figure output format

.. py:function:: fig_size:

    `tuple` (default = ``(10, 6)``)

    .. ``automatic``, `tuple` (default = ``automatic``)

    Tuple of 2 parameters pertaining to the size of the figures

.. py:function:: fig_dpi:

    ``int`` (default = ``150``)

    Figure dpi

.. py:function:: line_length:

    ``float`` (default = ``5.``)

    Length of image scale indicator

.. py:function:: arrow_length:

    ``float`` (default = ``0.15``)

    Length of compass arrow in image plots


