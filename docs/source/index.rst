.. LEOSatpy documentation master file, created by
   sphinx-quickstart on Wed May 31 15:00:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _ckoir: https://www.astro.uantof.cl/research/observatorios/ckoirama-observatory/

.. |ckoir| replace:: Ckoirama Observatory

.. _ckoir_2: https://www.astro.uantof.cl/research/observatorios/ckoirama-observatory/

.. |ckoir_2| replace:: *Ckoirama Observatory*

.. _ctio: https://noirlab.edu/science/programs/ctio/telescopes/smarts-consortium/smarts-consortium-09-meter-telescope

.. |ctio| replace:: Small and Moderate Aperture Research Telescope System

.. _ctio_4m: https://noirlab.edu/science/programs/ctio/telescopes/victor-blanco-4m-telescope

.. |ctio_4m| replace:: Víctor M. Blanco 4-meter Telescope

.. _dk154: https://www.eso.org/public/teles-instr/lasilla/danish154/

.. |dk154| replace:: 1.54-metre Danish telescope

.. _dk154_2: https://www.eso.org/public/teles-instr/lasilla/danish154/

.. |dk154_2| replace:: *1.54-metre Danish telescope*

.. _spm: https://www.astrossp.unam.mx/es/

.. |spm| replace:: Observatorio Astronómico Nacional

.. _ouka: https://moss-observatory.org/

.. |ouka| replace:: MOSS telescope

.. _cbnuo: https://www.chungbuk.ac.kr/site/english/main.do

.. |cbnuo| replace:: Chungbuk National University Observatory

.. _ca123: https://www.caha.es/CAHA/Telescopes/1.2m.html

.. |ca123| replace:: 1.23-metre telescope

.. _fut: https://phys.au.dk/en/news/item/artikel/fut-det-fjernstyrede-undervisningsteleskop-er-klar-til-de-foerste-gymnasieklasser-1

.. |fut| replace:: Fjernstyrede Undervisnings Teleskop

.. _mt-kent: https://science.desi.qld.gov.au/research/capability-directory/mount-kent-observatory

.. |mt-kent| replace:: Mt. Kent Observatory


========
LEOSatpy
========

**LEOSatpy** (Low Earth Orbit satellite python) is an end-to-end pipeline to process and analyse
satellite trail observations from different telescopes.

.. note::

   When publishing data processed and analysed with LEOSatpy, please :ref:`cite LEOSatpy <Citing LEOSatpy>`.

.. figure:: ./figs/home_leosat_two_examples.png
   :width: 750px

   *Two example of satellite observations reduced and analysed with LEOSatpy:
   V band observation of OneWeb-0108 taken with the* |dk154_2|_ *at the La Silla Observatory, Chile,
   and Sloan-r' band observation of Starlink-5464 from the* |ckoir_2|_ *of the University of Antofagasta, Chile.
   Used comparison stars are marked with red circles.
   The position and path predicted by the TLE orbital elements are shown in blue.*

Contents
========

.. toctree::
   :maxdepth: 1

   Home <self>
   installation
   example
   usage
   configfile
   contributing
   citing
   acknowledgements
   GitHub <https://github.com/CLEOsat-group/leosatpy>

----

The pipeline is written in Python 3 and provides the following functionalities:

===========================  ==========================================================================
Module                       Function
===========================  ==========================================================================
``reduceSatObs``             Full reduction of raw-FITS images including bias, dark, and flat reduction.
``calibrateSatObs``          WCS calibration, i.e. plate solving, using `GAIA DR3 <https://ui.adsabs.harvard.edu/abs/2020yCat.1350....0G/abstract>`_ positions, obtained via the `Astroquery <https://astroquery.readthedocs.io/en/latest/#>`_ tool.
``analyseSatObs``            Satellite trail(s) detection and aperture photometry using
                             comparison stars from the `GSC v2.4.3 <https://ui.adsabs.harvard.edu/#abs/2008AJ....136..735L>`_ catalog.
===========================  ==========================================================================

Supported Telescopes
====================

.. figure:: ./figs/home_cleosat_network.png
   :width: 750px
   :align: center

   *Telescopes currently participating in the CLEOSat observation network.*


LEOSatpy currently supports the following telescopes:


* 0.6-metre Chakana telescope at the |ckoir|_ of the Universidad de Antofagasta, Antofagasta, Chile.
* 0.9-metre |ctio|_ (SMARTS) at the Cerro Tololo Inter-american Observatory (CTIO), Chile.
* |ctio_4m|_ at the Cerro Tololo Inter-american Observatory (CTIO), Chile.
* |dk154|_ at the La Silla Observatory, Chile.
* 0.28-metre DDOTI (Deca-Degree Optical Transient Imager) telescopes at the |spm|_ (OAN) in Sierra San Pedro Martír (SPM), Baja California, México.
* 0.5-metre |ouka|_ at the Oukaïmeden Observatory, Morocco.
* 0.6-metre telescope of the |cbnuo|_ in Jincheon, South Korea.
* |ca123|_ at the Calar Alto Observatory, Spain.
* **(Work in Progress)** 0.6-metre |fut|_ (FUT) from Aarhus University at the |mt-kent|_ Observatory, Australia.


.. note::

    If you want your telescope added to the list, please contact
    `Jeremy Tregloan-Reed <jeremy.tregloan-reed@uda.cl>`_.


----------------


Copyright notice:
=================

The LEOSat Python package is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
version 3 of the License.

The LEOSat Python package is distributed in the hope that it will be useful, but without
any warranty; without even the implied warranty of merchantability or fitness for a
particular purpose. See the GNU General Public
`LICENSE <https://github.com/CLEOsat-group/leosatpy/blob/master/LICENSE>`_ file for the precise terms and conditions..

You should have received a copy of the GNU General Public License along with the LEOSat
Python package. If not, see http://www.gnu.org/licenses/.
