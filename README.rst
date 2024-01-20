.. Define variables

.. _ckoir: https://www.astro.uantof.cl/research/observatorios/ckoirama-observatory/

.. |ckoir| replace:: Ckoirama Observatory

.. _ctio: https://noirlab.edu/science/programs/ctio/telescopes/smarts-consortium/smarts-consortium-09-meter-telescope

.. |ctio| replace:: Small and Moderate Aperture Research Telescope System

.. _ctio_4m: https://noirlab.edu/science/programs/ctio/telescopes/victor-blanco-4m-telescope

.. |ctio_4m| replace:: Víctor M. Blanco 4-meter Telescope

.. _dk154: https://www.eso.org/public/teles-instr/lasilla/danish154/

.. |dk154| replace:: 1.54-metre Danish telescope

.. _spm: https://www.astrossp.unam.mx/es/

.. |spm| replace:: Observatorio Astronómico Nacional

.. _ouka: https://moss-observatory.org/

.. |ouka| replace:: MOSS telescope

.. _cbnuo: https://www.chungbuk.ac.kr/site/english/main.do

.. |cbnuo| replace:: Chungbuk National University Observatory

.. _ca123: https://www.caha.es/CAHA/Telescopes/1.2m.html

.. |ca123| replace:: 1.23-metre telescope

.. |stars| image:: https://img.shields.io/github/stars/CLEOsat-Group/leosatpy?style=social
    :alt: GitHub Repo stars
    :target: https://github.com/CLEOsat-group/leosatpy

.. |watch| image:: https://img.shields.io/github/watchers/CLEOsat-Group/leosatpy?style=social
    :alt: GitHub watchers
    :target: https://github.com/CLEOsat-group/leosatpy

.. |pypi| image:: https://img.shields.io/pypi/v/leosatpy
    :alt: PyPI
    :target: https://pypi.org/project/leosatpy/

.. |python| image:: https://img.shields.io/pypi/pyversions/leosatpy
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/leosatpy/

.. |release| image:: https://img.shields.io/github/v/release/CLEOsat-Group/leosatpy?include_prereleases
    :alt: GitHub release (latest SemVer including pre-releases)
    :target: https://github.com/CLEOsat-group/leosatpy

.. |last-commit| image:: https://img.shields.io/github/last-commit/CLEOsat-Group/leosatpy
    :alt: GitHub last commit
    :target: https://github.com/CLEOsat-group/leosatpy

.. |license| image:: https://img.shields.io/github/license/CLEOsat-Group/leosatpy
    :alt: GitHub
    :target: https://github.com/CLEOsat-group/leosatpy/blob/master/LICENSE

.. |rtd| image:: https://readthedocs.org/projects/leosatpy/badge/?version=latest
    :target: https://leosatpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |zenodo| image:: https://zenodo.org/badge/526672685.svg
    :target: https://zenodo.org/badge/latestdoi/526672685
    :alt: Zenodo

..
    |stars| |watch|

LEOSatpy
========

.. badges

|pypi| |python| |release| |last-commit| |license| |rtd| |zenodo|

**LEOSatpy** (Low Earth Orbit satellite python) is an end-to-end pipeline to process and analyse
satellite trail observations from various telescopes.

The pipeline is written in Python 3 and provides the following functionalities:

===========================  ==========================================================================
Module                       Function
===========================  ==========================================================================
``reduceSatObs``             Full reduction of raw-FITS images including bias, dark, and flat reduction.
``calibrateSatObs``          WCS calibration, i.e. plate solving, using `GAIA DR3 <https://ui.adsabs.harvard.edu/abs/2020yCat.1350....0G/abstract>`_ positions, obtained via the `Astroquery <https://astroquery.readthedocs.io/en/latest/#>`_ tool.
``analyseSatObs``            Satellite trail(s) detection and aperture photometry using
                             comparison stars from the `GSC v2.4.3 <https://ui.adsabs.harvard.edu/#abs/2008AJ....136..735L>`_ catalog.
===========================  ==========================================================================

The full documentation for LEOSatpy can be found `here <http://leosatpy.readthedocs.io/>`_.

LEOSatpy is distributed under the GNU General Public License v3. See the
`LICENSE <https://github.com/CLEOsat-group/leosatpy/blob/master/LICENSE>`_ file for the precise terms and conditions.

Currently supported telescopes:
    * 0.6-metre Chakana telescope at the |ckoir|_ of the Universidad de Antofagasta, Antofagasta, Chile.
    * 0.9-metre |ctio|_ (SMARTS)
      at the Cerro Tololo Inter-american Observatory (CTIO), Chile.
    * |ctio_4m|_ at the Cerro Tololo Inter-american Observatory (CTIO), Chile.
    * |dk154|_ at the La Silla Observatory, Chile.
    * 0.28-metre DDOTI (Deca-Degree Optical Transient Imager) telescopes at the |spm|_ (OAN) in Sierra San Pedro Martír (SPM), Baja California, México.
    * 0.5-metre |ouka|_ at the Oukaïmeden Observatory, Morocco.
    * 0.6-metre telescope of the |cbnuo|_ in Jincheon, South Korea.
    * |ca123|_ at the Calar Alto Observatory, Spain.


.. note::

    If you want your telescope added to the list, please contact
    `Jeremy Tregloan-Reed <jeremy.tregloan-reed@uda.cl>`_.


How to use LEOSatpy
-------------------

The LEOSatpy pipeline is written for use with Python >=3.9.
To avoid unnecessary interference with the Python installed on the system, it is recommended to create a new Python environment
to run LEOSatpy, using for example `conda <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

..
    To run LEOSatpy on a machine with a different version of python, it is recommended to use
    LEOSatpy with a Conda environment.
    This allows to run the package without interfering directly with the system.

    To install Conda follow the instructions `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

A new conda environment can be created and activated with:

.. code-block:: sh

    $ conda create -n leosatpy_env python=3.9
    $ conda activate leosatpy_env


Installation
^^^^^^^^^^^^


LEOSatpy is available on `PyPI <https://pypi.org/project/leosatpy/>`_, and can be installed using pip:

.. code:: bash

    $ (leosatpy_env) pip install leosatpy

Alternatively, the latest release of LEOSatpy is also available from the `GitHub repository <https://github.com/CLEOsat-group/leosatpy>`_.


1. Clone the repository using git:
    .. dummy comment

.. code-block:: sh

    $ (leosatpy_env) git clone https://github.com/CLEOsat-group/leosatpy.git

2. Download the zip file from the GitHub repository:
    Navigate to the main page of the repository. Click on the "Code" button, then click "Download ZIP".


Once cloned or downloaded and extracted, LEOSatpy can be installed from anywhere by typing:

.. code:: bash

    $ (leosatpy_env) pip install -e PATH/TO/CLONED/GITHUB

or by navigating to the downloaded folder:

.. code-block:: sh

    $ (leosatpy_env) cd PATH/TO/CLONED/GITHUB

and using the following command in the terminal:

.. code:: bash

    $ (leosatpy_env) python setup.py install


The successful installation of LEOSatpy can be tested by trying to access the help or the version of LEOSatpy via:

.. code:: bash

    $ (leosatpy_env) reduceSatObs --help

    $ (leosatpy_env) reduceSatObs --version

If no error messages are shown, LEOSatpy is most likely installed correctly.


Running LEOSatpy
^^^^^^^^^^^^^^^^


Prerequisites
"""""""""""""

**1. Configuration**

LEOSatpy comes with a configuration file, called `leosatpy_config.ini`, containing an extensive list of parameter
that can be adjusted to modify the behaviour of LEOSatpy.

.. important::

    Upon the first execution, a copy of the leosatpy configuration file is placed in the ``/home/user`` directory.
    Please modify the file as required and re-run the program.

By default, information and results for each dataset are stored in a .csv file located in the ``/home/user`` directory.
The location and name of this file can be changed by modifying the following lines in the `leosatpy_config.ini`:

.. code-block::

    RESULT_TABLE_PATH = '~'
    RESULT_TABLE_NAME = 'results_leosatpy.csv'

**2. Folder structure**

Although there is some degree of freedom in the nomenclature and structuring of the folder,
it is highly recommended to adopt the following folder layout:

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
Possible formats are, e.g., 20221110, 2022-11-20, tel_20221011_satxy, 2022-11-26_satxy_obs1, etc.

.. note::

    The program can detect and handle if the name of the folder does not corresponds to the observation date.
    However, the difference in date should not exceed 7 days. For example, data observed on 2022-11-11 UTC
    might be located in a folder named 2022-11-10. <-- This is detected.

It is also recommended to separate the raw calibration files, i.e., bias, darks, and flats from the science observation files
and place them into separate folder, named accordingly ``/bias``, ``/darks``, ``/flats``, and ``science/raw``, respectively.

Once all programs have been executed, the final folder structure should look like this:

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

.. attention::

    To prevent unexpected behaviour during the program execution, please also check and make sure that:

    * the raw FITS-files contain data
    * FITS-header keywords (e.g., `IMAGETYP` of bias, flats, or science files) are correctly labeled
    * corresponding raw FITS calibration images are available (e.g., binning, exposure time, filter)


LEOSatpy is now ready for use.


Reduction
"""""""""

The reduction of all raw FITS-files in a folder can be performed via the following line:

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs PATH/TO/DATA

LEOSatpy also accepts relative paths and multiple inputs, for example:

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs ../Telescope-Identifier/YYYY-MM-DD/

    $ (leosatpy_env) reduceSatObs PATH/TO/DATA/NIGHT_1 PATH/TO/DATA/NIGHT_2

To reduce all data from a telescope at once with:

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs PATH/TO/TELESCOPE/DATA

.. hint::

    The usage of partial and multiple inputs as shown above also works for the other programs in the package.



Astrometric calibration
"""""""""""""""""""""""

To apply the astrometric calibration type:

.. code-block:: sh

    $ (leosatpy_env) calibrateSatObs PATH/TO/DATA


Satellite trail detection and analysis
""""""""""""""""""""""""""""""""""""""

To run the satellite detection and analysis on all files in the input type:

.. code-block:: sh

    $ (leosatpy_env) analyseSatObs PATH/TO/DATA



Citing LEOSatpy
---------------

When publishing data processed and analysed with LEOSatpy, please cite:

::

    Adam et al. (2023) (in preparation). "Estimating the impact to astronomy from the Oneweb satellite constellation using multicolour observations". https://doi.org/10.5281/zenodo.8012131
    Software pipeline available at https://github.com/CLEOsat-group/leosatpy.

Acknowledgements
----------------

Alongside the packages listed in the ``requirements.txt``, this project uses workflows and code adopted from the following packages:

* `Astrometry <https://github.com/lukaswenzl/astrometry>`_ under the GPLv3 License, Lukas Wenzl (2022), `Zenodo <https://doi.org/10.5281/zenodo.6462441>`_
* `AutoPhOT <https://github.com/Astro-Sean/autophot>`_ under the GPLv3 License, Brennan & Fraser (2022), `NASA ADS <https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B/abstract>`_
* `Ccdproc <https://ccdproc.readthedocs.io/en/stable/index.html>`_, an Astropy package for image reduction (`Craig et al. 2023 <https://doi.org/10.5281/zenodo.593516>`_).

.. * `reduceccd <https://github.com/rgbIAA/reduceccd/tree/master>`_ under the BSD-3-Clause license
.. * `wht_reduction_scripts <https://github.com/crawfordsm/wht_reduction_scripts>`_ under the BSD-3-Clause license


The authors of these packages and code are gratefully acknowledged.

Special thanks go out to the following people for their ideas and contributions to the development
of the LEOSat Python package:

* `Jeremy Tregloan-Reed <jeremy.tregloan-reed@uda.cl>`_, Universidad de Atacama
* `Eduardo Unda-Sanzana <eduardo.unda@uamail.cl>`_, Universidad de Antofagasta
* `Edgar Ortiz <ed.ortizm@gmail.com>`_, Universidad de Antofagasta
* `Maria Isabel Romero Colmenares <maria.romero.21@alumnos.uda.cl>`_, Universidad de Atacama
* `Sangeetha Nandakumar <an.sangeetha@gmail.coml>`_, Universidad de Atacama

The project would not have been possible without the help of everyone who contributed.



Feedback, questions, comments?
------------------------------

LEOSatpy is under active development and help with the development of new functionalities
and fixing bugs is very much appreciated.
In case you would like to contribute, feel free to fork the
`GitHub repository <https://github.com/CLEOsat-group/leosatpy>`_ and to create a pull request.

If you encounter a bug or problem, please `submit a new issue on the GitHub repository
<https://github.com/CLEOsat-group/leosatpy/issues>`_ providing as much
detail as possible (error message, operating system, Python version, etc.).

If you have further feedback, questions or comments you can also send an e-mail to
`Jeremy Tregloan-Reed <jeremy.tregloan-reed@uda.cl>`_, or `Christian Adam <christian.adam84@gmail.com>`_.


Author
------

`Christian Adam <christian.adam84@gmail.com>`_,
Centro de Investigación, Tecnología, Educación y Vinculación Astronómica (CITEVA), Universidad de Antofagasta,
Antofagasta, Chile
