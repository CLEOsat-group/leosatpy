
Installation
============

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


Installing LEOSatpy
-------------------

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

Testing LEOSatpy
----------------

The successful installation of LEOSatpy can be tested by trying to access the help or the version of LEOSatpy via:

.. code:: bash

    $ (leosatpy_env) reduceSatObs --help

    $ (leosatpy_env) reduceSatObs --version

If no error messages are shown, LEOSatpy is most likely installed correctly.



-----

As a next step and before using LEOSatpy, it is recommended to edit the :ref:`configuration file <Configuration file>`.

.. attention::

    Upon the first-time execution, a copy of the configuration file is placed in the ``/home/user`` directory.
    Please modify the configuration as required before running the program on the data.
