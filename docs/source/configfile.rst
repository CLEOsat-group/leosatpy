
Configuration file
==================


The configuration file with the available parameters for LEOSatpy is shown below:

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

*H*\ :sub:`orb`,
`ccdproc <https://ccdproc.readthedocs.io/en/latest/>`_

Name of fitting method to use

General options
---------------

.. _result_table_path:

.. py:function:: result_table_path:

    ``str`` (default = ``/home/username/``)

    :file:`/home/username/`

.. py:function:: result_table_name:

    ``str`` (default = ``results_leosatpy.csv``)

    :file:`/home/username/`

.. py:function:: roi_table_name:

    ``str`` (default = ``leosatpy_regions_of_interest.csv``)

    .. warning::
        Duplicate file name entries are not supported at the moment.


Reduction options
-----------------

.. py:function:: bias_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``,

.. py:function:: dark_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``,

.. py:function:: flat_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``,

.. py:function:: flat_dark_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``,

.. py:function:: overscan_correct:

   ``True``, ``False`` (default = ``False``)

   If ``True``,

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

    .. :file:`/home/username/`

.. py:function:: combine_method_dark:

    ``average``, ``median``, ``sum`` (default = ``average``)

    .. :file:`/home/username/`

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



.. py:function:: source_min_no:

    ``int`` (default = ``5``)


.. py:function:: source_max_no:

    ``int`` (default = ``1000``)


.. py:function:: fwhm_init_guess:

    ``float`` (default = ``5.``)


.. py:function:: fwhm_lim_min:

    ``float`` (default = ``1.``)


.. py:function:: fwhm_lim_max:

    ``float`` (default = ``30.``)


.. py:function:: threshold_value:

    ``float`` (default = ``10.``)


.. py:function:: threshold_value_lim:

    ``float`` (default = ``3.``)


.. py:function:: threshold_fudge_factor:

    ``float`` (default = ``5.``)


.. py:function:: threshold_fine_fudge_factor:

    ``float`` (default = ``0.1``)


.. py:function:: sigmaclip_fwhm_sigma:

    ``float`` (default = ``3.``)


.. py:function:: isolate_sources_init_sep:

    ``float`` (default = ``20.``)


.. py:function:: isolate_sources_fwhm_sep:

    ``float`` (default = ``5.``)


.. py:function:: max_func_iter:

    ``int`` (default = ``50``)


.. py:function:: fitting_method:

    ``str`` (default = ``least_square``)

    analytical function fitting using `LMFIT <https://lmfit.github.io/lmfit-py/fitting.html>`_
    We can accept a limited number of methods from `here <https://lmfit.github.io/lmfit-py/fitting.html>`_.

.. py:function:: default_moff_beta:

    ``float`` (default = ``4.765``)

    `Trujillo et al. (2001) <https://doi.org/10.1046/j.1365-8711.2001.04937.x>`_


(e.g. :math:`f>5\sigma`)

WCS calibration options
-----------------------


.. py:function:: ref_catalog_mag_lim:

    ``median`` (default = ``results_leosatpy.csv``)

    :file:`/home/username/`

.. py:function:: max_wcs_func_iter:

    ``int`` (default = ``10``)


.. py:function:: ref_sources_max_no:

    ``int`` (default = ``2500``)


.. py:function:: max_no_obs_sources:

    ``-1``, ``int`` (default = ``-1``)


.. py:function:: distance_bin_size:

    ``float`` (default = ``1.``)


.. py:function:: ang_bin_size:

    ``float`` (default = ``0.1``)



.. py:function:: match_radius:

    ``fwhm``, ``float`` (default = ``fwhm``)


.. py:function:: min_source_no_convergence:

    ``int`` (default = ``5``)


.. py:function:: threshold_convergence:

    ``float`` (default = ``0.95``)



Satellite detection & analysis options
---------------------------------------

.. py:function:: parallel_trail_select:

    ``int`` (default = ``0``)

    :file:`/home/username/`


.. py:function:: max_distance:

    ``int`` (default = ``2``)


.. py:function:: n_trails_max:

    ``int`` (default = ``1``)


.. py:function:: rho_bin_size:

    ``float`` (default = ``0.5``)


.. py:function:: theta_bin_size:

    ``float`` (default = ``0.02``)


.. py:function:: theta_sub_win_size:

    ``float`` (default = ``3.``)


.. py:function:: rho_sub_win_res_fwhm:

    ``float`` (default = ``6.``)


.. py:function:: trail_params_fitting_method:

    ``str`` (default = ``least_square``)


.. py:function:: num_std_min:

    ``int`` (default = ``5``)


.. py:function:: num_std_max:

    ``int`` (default = ``500``)



.. py:function:: aper_rad:

    ``float`` (default = ``1.7``)


.. py:function:: inf_aper_rad:

    ``float`` (default = ``2.5``)


.. py:function:: aper_start:

    ``float`` (default = ``0.1``)


.. py:function:: aper_stop:

    ``float`` (default = ``5``)


.. py:function:: aper_step_size:

    ``float`` (default = ``0.1``)


.. py:function:: rskyin:

    ``float`` (default = ``2.``)


.. py:function:: rskyout:

    ``float`` (default = ``3.``)


.. py:function:: dt_step_size:

    ``float`` (default = ``0.01``)


Figure options
---------------

.. py:function:: fig_size:

    `tuple` (default = ``(10, 6)``)

    .. ``automatic``, `tuple` (default = ``automatic``)

    Tuple of 2 parameters pertaining to the size of the figures

.. py:function:: fig_type:

    ``png``, ``pdf`` (default = ``png``)


.. py:function:: fig_dpi:

    ``int`` (default = ``150``)


.. py:function:: line_length:

    ``float`` (default = ``5.``)


.. py:function:: arrow_length:

    ``float`` (default = ``0.15``)



