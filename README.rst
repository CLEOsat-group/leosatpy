========
LEOSatpy
========

**LEOSatpy** (Low Earth Orbit satellite python) is an end-to-end pipeline to process and analyse
satellite trail observations from different telescopes.

The programs in the LEOSatpy package are written for python 3.9 (recommended) and 3.10.
To run LEOSatpy on a machine with a different version of python, it is recommended to use
LEOSatpy with a Conda environment.
This allows to run the package without interfering with the system.

To install Conda follow the instructions
`here <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.
Once installed, a conda environment running a specific version of Python can be created and activated with:

.. code-block:: sh

    $ conda create -n myenv python=3.9
    $ conda activate myenv


The full documentation for **LEOSatpy** can be found `here <https://docs.readthedocs.io/en/stable/tutorial/>`_.


LEOSatpy is distributed under the GNU General Public License v3. See the
`LICENSE <https://github.com/CLEOsat-group/leosatpy/blob/master/LICENSE>`_ file for the precise terms and conditions.

-----------------------
Package Functionalities
-----------------------

^^^^^^^^^^^^^^^^^^^^
Content
^^^^^^^^^^^^^^^^^^^^

The important files for the process in this directory:

===========================  ==========================================================================
Program                      Function
===========================  ==========================================================================
``reduceSatObs.py``          Perform reduction of satellite observations.
``calibrateSatObs.py``       Perform astrometric calibration using GAIA eDR3 positions.
``analyseSatObs.py``         Detect satellite trail(s) and perform aperture photometry using
                             comparison stars from the GSC 2.4 catalog.
===========================  ==========================================================================

^^^^^^^^^^^^^^^^^^^^
Supported Telescopes
^^^^^^^^^^^^^^^^^^^^

Observations from the following telescopes are currently supported:

* Danish 1.54-metre telescope at La Silla, Chile.
* 0.6-metre telescope of the Chungbuk National University Observatory in Jincheon, South Korea
* Chakana 0.6-metre telescope at the Ckoirama observatory, Antofagasta, Chile.
* 1.23-metre telescope at Calar Alto, Spain

.. put links to telescopes

The positions and magnitudes of the comparison stars are collected using the
`WebServices for Catalog Access <https://outerspace.stsci.edu/display/GC/WebServices+for+Catalog+Access>`_
to the Guide Star Catalog(s).

.. `Link <Feedback, comments, questions?_>`_

-------------------
How to use LEOSatpy
-------------------

^^^^^^^^^^^^
Installation
^^^^^^^^^^^^

To run the `leosatpy` programs, download or clone the repository to your local machine

.. code-block:: sh

    $ (myenv) git clone https://github.com/CLEOsat-group/leosatpy.git

and navigate to the folder location of the cloned package files:

.. code-block:: sh

    $ (myenv) cd path/to/cloned/github

**INSTALLATION: To be implemented/written**

""""""""""""
Requirements
""""""""""""

The LEOSatpy programs are written with python 3.9 and tested with python 3.9 and 3.10.

Alongside the standard python modules such as numpy, pandas, scipy, or matplotlib,
LEOSatpy also requires a number of additional modules to work, e.g. ccdproc, astropy, or photutils.

All packages required to run LEOSatpy can be install at once with:

.. code-block:: sh

    $ (myenv) pip install -r requirements.txt


^^^^^^^^^^^^^^^^
Running LEOSatpy
^^^^^^^^^^^^^^^^

"""""""""""""
Prerequisites
"""""""""""""

**1. Configuration**

The LEOSatpy package comes with a configuration file, called `leosatpy_config.ini`.

..    This file allows to change a number of parameter used during the reduction, calibration and analysis.
    Among these are the location and name of the result table holding all collected information and analysis results.

By default the results are saved in the ``/home/user`` directory.
To change the location and name open the configuration file and change the following lines:

::

    RESULT_TABLE_PATH = '~'
    RESULT_TABLE_NAME = 'results_LEOSat.csv'

**2. Folder structure**

Although there is some degree of freedom in the nomenclature and structuring of the folder,
it is recommended to follow the folder layout given below:

.. code-block::

    .
    └── Telescope-Identifier <- free naming
        ├── YYYY-MM-DD <- mandatory format
        │   ├── bias
        │   ├── flats
        │   ├── darks
        │   └── science_data <- free naming
        │       └── raw <- optional, but recommended
        ├── YYYY-MM-DD
        └── YYYY-MM-DD

The only mandatory requirement is the naming of the main folder
containing the observations of a single night.
This folder should only contain the date of observation using the following format: ``YYYY-MM-DD``.

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
        │       └── raw
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


We are now ready to run LEOSatpy.

"""""""""
Reduction
"""""""""

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


..    During the reduction the following steps are performed:

        * Image registration and validation
        * Master calibration file creation
        * Removal of instrumental signatures to create and save the reduced FITS-image(s)
        * Save results to result table.

"""""""""""""""""""""""
Astrometric calibration
"""""""""""""""""""""""

To apply the astrometric calibration type:

.. code-block:: sh

    $ (myenv) python calibrateSatObs.py [path_to_data]

..    During the astrometric calibration the following steps are performed:

        * Registration and validation of the reduced FITS-files
        * 2D background estimation and source detection
        * Determination of the pixel scale and detector rotation angle by comparing the detected sources with precise positions from the GAIA eDR3 catalog
        * Update the FITS-files World Coordinate System (WCS) with found transformation.
        * Save results to result table

""""""""""""""""""""""""""""""""""""""
Satellite trail detection and analysis
""""""""""""""""""""""""""""""""""""""

To run the satellite detection and analysis on all files in the input type:

.. code-block:: sh

    $ (myenv) python analyseSatObs.py [path_to_data]

..  During the analysis the following steps are performed:

    * Registration and validation of the calibrated FITS-files
    * `Xu et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015PatRe..48.4012X/abstract>`_
    * Save results to result table

----
ToDo
----

.. * Memory and speed optimizations
* Add full pre-processing check of files in input path before reduction
* Add align and combine to reduction to make it more general


---------------
Citing LEOSatpy
---------------

When publishing data processed and analysed with LEOSatpy, please cite `TBW`

----------------
Acknowledgements
----------------

* funding
* used code sources
* etc.

------------------------------
Feedback, comments, questions?
------------------------------

Please send an e-mail to: `CLEOSat-Group <christian.adam84@gmail.com>`_.

^^^^^^
Author
^^^^^^

Christian Adam
