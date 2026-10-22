
Example analysis
================

- complete example, download from Zenodo and GitHub
- folder leosatpy_example_data
- raw data from a LEO satellite, use Korean telescope data with full trail. give image details, exposure time, date, time, location, filter, etc.
- show directory structure before and after reduction
- show commands to run reduction, calibration, analysis (code blocks, terminal commands, maybe as gif?)
- show example plots for each step
- `DS9 <https://sites.google.com/cfa.harvard.edu/saoimageds9>`_.



Reduction
---------

.. code-block:: sh

    $ (leosatpy_env) reduceSatObs PATH/TO/DATA

Add comparison images between raw and reduced data.

Calibration
-----------

.. code-block:: sh

    $ (leosatpy_env) calibrateSatObs PATH/TO/DATA

Analysis
---------


.. code-block:: sh

    $ (leosatpy_env) analyseSatObs PATH/TO/DATA

