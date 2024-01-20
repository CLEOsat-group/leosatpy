
Usage instructions
==================

- :ref:`Running LEOSatpy`
- :ref:`Reduction in a nutshell`
- :ref:`WCS calibration in a nutshell`
- :ref:`Satellite trail analysis in a nutshell`

Running LEOSatpy
----------------

Analysing satellite data with LEOSatpy is straightforward and only requires a few commands to be entered in the terminal.
Upon the first-time execution, a copy of the configuration file (:file:`leosatpy_config.ini`) is placed in the ``/home/username/`` directory.
The parameters in the configuration file can then be adjusted with a text editor (see :ref:`Configuration file`).

LEOSatpy offers some degree of freedom in the nomenclature and structuring of the folder.
However, it is recommended to follow the folder layout given below:

.. code-block::

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

    The main folder name should contain the date of observation either in the format: ``YYYY-MM-DD``, or ``YYYYMMDD``.

    LEOSatpy automatically selects the search path for the calibration data based on the observation date from the science FITS-header
    and the names of folder in the input path.
    Possible formats are, e.g., ``20221110``, ``2022-11-20``, ``tel_20221011_satxy``, ``2022-11-26_satxy_obs1``, etc.

.. .. note::

        The program can detect and handle if the name of the folder does not corresponds to the observation date.
        However, the difference in date should not exceed 7 days. For example, data observed on 2022-11-11 UTC
        might be located in a folder named 2022-11-10. <-- This is detected.

It is also recommended to separate the raw calibration files, i.e., bias, darks, and flats from the science observation files
and place them into separate folder, named accordingly :file:`/bias`, :file:`/darks`, :file:`/flats`, and :file:`science/raw`, respectively.


LEOSatpy is now ready for use.



.. link to example

.. .. admonition:: Multiple Hints

       -  typing :code:`reduceSatObs --help`
       -

Reduction in a nutshell
-----------------------

The reduction of all raw FITS-files in a folder can be performed via the following line:

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs PATH/TO/TELESCOPE/DATA/NIGHT_1

LEOSatpy also accepts multiple inputs:

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs PATH/TO/DATA/NIGHT_1 PATH/TO/DATA/NIGHT_2

and allows to reduce all data from a telescope at once with:

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs PATH/TO/TELESCOPE

.. note::

    Relative paths are also acceptable, e.g., :code:`reduceSatObs ../Telescope-Identifier/YYYY-MM-DD/`.

.. .. attention::
    
        To prevent unexpected behaviour during execution, please also check that:

        * the raw FITS-files contain data
        * FITS-header keywords (e.g., `IMAGETYP` of bias, flats, or science files) are correctly labeled
        * corresponding raw FITS calibration images are available (e.g., binning, exposure time, filter)

.. tip::

    The usage of partial and multiple inputs as shown above also works for the other programs in the package.

Image registration and validation

.. important::

    LEOSatpy will not overwrite any original data.

Master calibration file creation

Removal of instrumental signatures to create and save the reduced FITS-image(s)

During the reduction the following steps are performed:

Save observation information to the result .csv table.

* Image registration and validation
* Master calibration file creation
* Removal of instrumental signatures to create and save the reduced FITS-image(s)
* Save observation information to the result .csv table.
* setting :ref:`result_table_path <General options>`, :ref:`result_table_name <General options>`

:ref:`combine_method_flat <Reduction options>`

OBJECT-files  FLAT, LAMP, DARK, BIAS

.. important::

    When finished, check the FITS-files that LEOSatpy writes to the subdirectories ``reduced`` and ``master_calibs``.


WCS calibration in a nutshell
-----------------------------

To apply the astrometric calibration on the reduced OBJECT-files, type:

.. code-block:: sh

    $ (leosatpy_env) calibrateSatObs PATH/TO/TELESCOPE/DATA/NIGHT_1


During the astrometric calibration the following steps are performed:

* Registration and validation of the reduced FITS-files
* 2D background estimation and source detection
* Determination of the pixel scale and detector rotation angle by comparing the detected sources with precise positions from the GAIA eDR3 catalog
* Update the FITS-files World Coordinate System (WCS) with found transformation.
* Save results to result table

.. important::

    Always check the FITS-files and figures that IRDAP writes to the subdirectories ``calibrated`` and ``figures``.



Satellite trail analysis in a nutshell
--------------------------------------

To run the satellite detection and analysis on all files in the input type:

.. code-block:: sh

    $ (leosatpy_env) analyseSatObs PATH/TO/TELESCOPE/DATA/NIGHT_1

During the analysis the following steps are performed:

* Registration and validation of the calibrated FITS-files
* `Xu et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015PatRe..48.4012X/abstract>`_
* Save results to result table

----

Once all programs have been executed, the directory should look like this:

.. code-block::

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

To better understand the input parameters, continue with the :ref:`configuration file <Configuration file>`.