
Installation
============

To run LEOSatpy the user needs to have Python 3.9 or 3.10 installed.

To run LEOSatpy on a machine with a different version of python, it is recommended to use
LEOSatpy with a Conda environment.
This allows to run the package without interfering with the system.

To install Conda follow the instructions
`here <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.
Once installed, a conda environment running a specific version of Python can be created and activated with:

.. code-block:: sh

    $ conda create -n myenv python=3.9
    $ conda activate myenv

How-To obtain LEOSatpy
---------------------

To run the programs contained in the `LEOSatpy` package, download or clone the repository to a folder of your choice on your local machine using
the following command

.. code-block:: sh

    $ (myenv) git clone https://github.com/CLEOsat-group/leosatpy.git

and navigate to the folder location of the cloned package files:

.. code-block:: sh

    $ (myenv) cd leosatpy

Installing LEOSatpy
-------------------

.. note::

    **PyPi and setup.py installation: Coming soon**

-----

As a next step and before using LEOSatpy, it is recommended to edit the :ref:`configuration file <Configuration file>`.

