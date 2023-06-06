
Usage instructions
==================

- :ref:`Running LEOSatpy`
- :ref:`Reduction in a nutshell`
- :ref:`WCS calibration in a nutshell`
- :ref:`Satellite trail analysis in a nutshell`

Running LEOSatpy
----------------

.. important::

    LEOSatpy will not overwrite any original data.

Although there is some degree of freedom in the nomenclature and structuring of the folder,
it is recommended to follow the folder layout given below:

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

The only requirement with regard to the name of the main folder is
that the folder name should contain the date of observation either in the format: ``YYYY-MM-DD``, or ``YYYYMMDD``.

The program will select the search path for the calibration data based on the obs date from the science data header
and the names of folder in the given path.
Possible formats are, e.g., ``20221110``, ``2022-11-20``, ``tel_20221011_satxy``, ``2022-11-26_satxy_obs1``, etc.

.. note::

    The program can detect and handle if the name of the folder does not corresponds to the observation date.
    However, the difference in date should not exceed 7 days. For example, data observed on 2022-11-11 UTC
    might be located in a folder named 2022-11-10. <-- This is detected.

It is also recommended to separate the raw calibration files from the science observation files
and place them into separate folder.

Once all programs have been executed, the directory should look like this:

.. code-block::

    .
    └── Telescope-Identifier
        ├── YYYY-MM-DD
        │   ├── bias
        │   ├── flats
        │   ├── darks
        │   ├── master_calibs
        │   └── science_data
        │       ├── auxiliary
        │       ├── calibrated
        │       ├── catalogs
        │       ├── figures
        │       │   └── Sat-ID
        │       ├── raw
        │       └── reduced
        ├── YYYY-MM-DD
        └── YYYY-MM-DD

.. attention::

    To prevent unexpected behaviour during execution, please also check that:

    * the raw FITS-files contain data
    * FITS-header keywords (e.g., `IMAGETYP` of bias, flats, or science files) are correctly labeled
    * corresponding raw FITS calibration images are available (e.g., binning, exposure time, filter)

LEOSatpy is now ready for use.

You can then adjust the parameters in the configuration file with a text
editor (see :ref:`Configuration file`)

.. link to example

.. admonition:: Multiple Hints

   -  typing :code:`reduceSatObs --help`
   -

Reduction in a nutshell
-----------------------

The reduction of all raw FITS-files in a folder can be performed via the following line:

.. code-block:: sh

    $ (myenv) python reduceSatObs.py [path_to_data]

For example:

.. code-block:: sh

    $ (myenv) python reduceSatObs.py ../Telescope-Identifier/YYYY-MM-DD/

To reduce data from multiple nights for example type:

.. code-block:: sh

    $ (myenv) python reduceSatObs.py [path_to_data_night_1] [path_to_data_night_2]

It is also possible to reduce all epochs of a telescope at once with:

.. code-block:: sh

    $ (myenv) python reduceSatObs.py [path_to_telescope_data]

.. note::

    The usage of partial and multiple inputs as shown above also works for the other programs in the package.


During the reduction the following steps are performed:

* Image registration and validation
* Master calibration file creation
* Removal of instrumental signatures to create and save the reduced FITS-image(s)
* Save observation information to the result .csv table.
* setting :ref:`result_table_path <General options>`



WCS calibration in a nutshell
-----------------------------

To apply the astrometric calibration type:

.. code-block:: sh

    $ (myenv) python calibrateSatObs.py [path_to_data]

During the astrometric calibration the following steps are performed:

* Registration and validation of the reduced FITS-files
* 2D background estimation and source detection
* Determination of the pixel scale and detector rotation angle by comparing the detected sources with precise positions from the GAIA eDR3 catalog
* Update the FITS-files World Coordinate System (WCS) with found transformation.
* Save results to result table


Satellite trail analysis in a nutshell
--------------------------------------

To run the satellite detection and analysis on all files in the input type:

.. code-block:: sh

    $ (myenv) python analyseSatObs.py [path_to_data]

During the analysis the following steps are performed:

* Registration and validation of the calibrated FITS-files
* `Xu et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015PatRe..48.4012X/abstract>`_
* Save results to result table
