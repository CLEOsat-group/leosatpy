# LEOSatpy

[![PyPI](https://img.shields.io/pypi/v/leosatpy)](https://pypi.org/project/leosatpy/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/leosatpy)](https://pypi.org/project/leosatpy/)
[![GitHub release (latest SemVer including pre-releases)](https://img.shields.io/github/v/release/CLEOsat-Group/leosatpy?include_prereleases)](https://github.com/CLEOsat-group/leosatpy)
[![GitHub last commit](https://img.shields.io/github/last-commit/CLEOsat-Group/leosatpy)](https://github.com/CLEOsat-group/leosatpy)
[![GitHub](https://img.shields.io/github/license/CLEOsat-Group/leosatpy)](https://github.com/CLEOsat-group/leosatpy/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/leosatpy/badge/?version=latest)](https://leosatpy.readthedocs.io/en/latest/?badge=latest)
[![Zenodo](https://zenodo.org/badge/526672685.svg)](https://zenodo.org/badge/latestdoi/526672685)

**LEOSatpy** (Low Earth Orbit satellite python) is an end-to-end
pipeline to process and analyse satellite trail observations from
various telescopes.

The pipeline is written in Python 3 and provides the following
functionalities:

| Module            | Function                                                                                                                                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reduceSatObs`    | Full reduction of raw-FITS images including bias, dark, and flat reduction.                                                                                                                                         |
| `calibrateSatObs` | WCS calibration, i.e. plate solving, using [GAIA DR3](https://ui.adsabs.harvard.edu/abs/2020yCat.1350....0G/abstract) positions, obtained via the [Astroquery](https://astroquery.readthedocs.io/en/latest/#) tool. |
| `analyseSatObs`   | Satellite trail(s) detection and aperture photometry using comparison stars from the [GSC v2.4.3](https://ui.adsabs.harvard.edu/#abs/2008AJ....136..735L) catalog.                                                  |

The full documentation for LEOSatpy can be found
[here](http://leosatpy.readthedocs.io/).

LEOSatpy is distributed under the GNU General Public License v3. See the
[LICENSE](https://github.com/CLEOsat-group/leosatpy/blob/master/LICENSE)
file for the precise terms and conditions.

Currently supported telescopes:

- 0.6-metre Chakana telescope at the 
[Ckoirama Observatory](https://www.astro.uantof.cl/research/observatorios/ckoirama-observatory/) 
of the Universidad de Antofagasta, Antofagasta, Chile.
- 0.9-metre 
[Small and Moderate Aperture Research Telescope System](https://noirlab.edu/science/programs/ctio/telescopes/smarts-consortium/smarts-consortium-09-meter-telescope)
(SMARTS) at the Cerro Tololo Inter-american Observatory (CTIO), Chile.
- [Víctor M. Blanco 4-meter Telescope](https://noirlab.edu/science/programs/ctio/telescopes/victor-blanco-4m-telescope) 
at the Cerro Tololo Inter-american Observatory (CTIO), Chile.
- 1.54-metre [Danish telescope](https://www.eso.org/public/teles-instr/lasilla/danish154/) 
at the La Silla Observatory, Chile.
- 0.28-metre DDOTI (Deca-Degree Optical Transient Imager) telescopes
  at the [Observatorio Astronómico Nacional](https://www.astrossp.unam.mx/es/) 
(OAN) in Sierra San Pedro Martír (SPM), Baja California, México.
- 0.5-metre [MOSS telescope](https://moss-observatory.org/) 
at the Oukaïmeden Observatory, Morocco.
- 0.6-metre telescope of the
[Chungbuk National University Observatory](https://www.chungbuk.ac.kr/site/english/main.do) in Jincheon, South Korea.
- [1.23-metre telescope](https://www.caha.es/CAHA/Telescopes/1.2m.html) 
at the Calar Alto Observatory, Spain.
- **(Work in Progress)** 0.6-metre [Fjernstyrede Undervisnings Teleskop](https://phys.au.dk/en/news/item/artikel/fut-det-fjernstyrede-undervisningsteleskop-er-klar-til-de-foerste-gymnasieklasser-1) 
(FUT) from Aarhus University at the [Mt. Kent Observatory](https://science.desi.qld.gov.au/research/capability-directory/mount-kent-observatory), Australia.

> [!NOTE]  
If you want your telescope added to the list, please contact [Jeremy Tregloan-Reed](mailto:jeremy.tregloan-reed@uda.cl).


## How to use LEOSatpy

The LEOSatpy pipeline is written for use with Python \>=3.9. To avoid
unnecessary interference with the Python installed on the system, it is
recommended to create a new Python environment to run LEOSatpy, using
for example [conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

A new conda environment can be created and activated with:

```sh
  $ conda create -n leosatpy_env python=3.9
  $ conda activate leosatpy_env
```

### Installation

LEOSatpy is available on [PyPI](https://pypi.org/project/leosatpy/), and can be installed using pip:

```bash
  $ (leosatpy_env) pip install leosatpy
```

Alternatively, the latest release of LEOSatpy is also available from the
[GitHub repository](https://github.com/CLEOsat-group/leosatpy).

1.  Clone the repository using git:  
    
    ```sh
      $ (leosatpy_env) git clone https://github.com/CLEOsat-group/leosatpy.git
    ```

2.  Download the zip file from the GitHub repository:  
    Navigate to the main page of the repository. Click on the "Code"
    button, then click "Download ZIP".

Once cloned or downloaded and extracted, LEOSatpy can be installed from
anywhere by typing:

```bash
  $ (leosatpy_env) pip install -e PATH/TO/CLONED/GITHUB
```

or by navigating to the downloaded folder:

```sh
  $ (leosatpy_env) cd PATH/TO/CLONED/GITHUB
```

and using the following command in the terminal:

```bash
  $ (leosatpy_env) python setup.py install
```

The successful installation of LEOSatpy can be tested by trying to
access the help or the version of LEOSatpy via:

```bash
  $ (leosatpy_env) reduceSatObs --help

  $ (leosatpy_env) reduceSatObs --version
```

If no error messages are shown, LEOSatpy is most likely installed correctly.

### Running LEOSatpy

#### Prerequisites

**1. Configuration**

LEOSatpy comes with a configuration file, called [leosatpy_config.ini](/leosatpy/leosatpy_config.ini), 
containing an extensive list of parameter that can be adjusted to modify the behaviour of LEOSatpy.

> [!IMPORTANT]  
Upon the first execution, a copy of the leosatpy configuration file is
placed in the `/home/user` directory. Please modify the file as required
and re-run the program.

By default, information and results for each dataset are stored in a
.csv file located in the `/home/user` directory. The location and name
of this file can be changed by modifying the following lines in the
[leosatpy_config.ini](/leosatpy/leosatpy_config.ini):

``` 
WORKING_DIR_PATH = '~'
RESULT_TABLE_NAME = 'results_leosatpy.csv'
```

**2. Folder structure**

Although there is some degree of freedom in the nomenclature and
structuring of the folder, it is highly recommended to adopt the
following folder layout:

``` 
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
```

The only requirement with regard to the name of the main folder is that
the folder name should contain the date of observation either in the
format: `YYYY-MM-DD`, or `YYYYMMDD`.

The program will select the search path for the calibration data based
on the obs date from the science data header and the names of folder in
the given path. Possible formats are, e.g., 20221110, 2022-11-20,
tel_20221011_satxy, 2022-11-26_satxy_obs1, etc.

> [!NOTE]
> The program can detect and handle if the name of the folder does not
correspond to the observation date. However, the difference in date
should not exceed 7 days. For example, data observed on 2022-11-11 UTC
might be located in a folder named 2022-11-10. \<-- This is detected.

It is also recommended to separate the raw calibration files, i.e.,
bias, darks, and flats from the science observation files and place them
into separate folder, named accordingly `/bias`, `/darks`, `/flats`, and
`science/raw`, respectively.

Once all programs have been executed, the final folder structure should
look like this:

``` 
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
```

> [!WARNING] 
> To prevent unexpected behaviour during the program execution, please also check and make sure that:
> -   the raw FITS-files contain data
> -   FITS-header keywords (e.g., <span class="title-ref">IMAGETYP</span>
    of bias, flats, or science files) are correctly labeled
> -   and the corresponding raw FITS calibration images are available 
> (e.g., binning, exposure time, filter).

LEOSatpy is now ready for use.

#### Reduction

The reduction of all raw FITS-files in a folder can be performed via the
following line:

```sh
  $ (leosatpy_env) reduceSatObs PATH/TO/DATA
```

LEOSatpy also accepts relative paths and multiple inputs, for example:

```sh
  $ (leosatpy_env) reduceSatObs ../Telescope-Identifier/YYYY-MM-DD/

```

or

```sh
  $ (leosatpy_env) reduceSatObs PATH/TO/DATA/NIGHT_1 PATH/TO/DATA/NIGHT_2
```

To reduce all data from a telescope at once with:

```sh
  $ (leosatpy_env) reduceSatObs PATH/TO/TELESCOPE/DATA
```

> [!TIP]  
The usage of partial and multiple inputs as shown above also works for
the other programs in the package.

#### Astrometric calibration

To apply the astrometric calibration type:

```sh
  $ (leosatpy_env) calibrateSatObs PATH/TO/DATA
```

#### Satellite trail detection and analysis

To run the satellite detection and analysis on all files in the input
type:

```sh
  $ (leosatpy_env) analyseSatObs PATH/TO/DATA
```

## Citing LEOSatpy

When publishing data processed and analysed with LEOSatpy, please cite:

    Adam et al. (2025) (in preparation). "Estimating the impact to astronomy from the Oneweb satellite constellation using multicolour observations". https://doi.org/10.5281/zenodo.8012131
    Software pipeline available at https://github.com/CLEOsat-group/leosatpy.

## Acknowledgements

Alongside the packages listed in the [requirements.txt](requirements.txt), this project
uses workflows and code adopted from the following packages:

-   [Astrometry](https://github.com/lukaswenzl/astrometry) under the GPLv3 License, Lukas Wenzl (2022), [Zenodo](https://doi.org/10.5281/zenodo.6462441)
-   [AutoPhOT](https://github.com/Astro-Sean/autophot) under the GPLv3 License, Brennan & Fraser (2022), [NASA ADS](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B/abstract)
-   [Ccdproc](https://ccdproc.readthedocs.io/en/stable/index.html), an Astropy package for image reduction ([Craig et al. 2023](https://doi.org/10.5281/zenodo.593516)).

The authors of these packages and code are gratefully acknowledged.

Special thanks go out to the following people for their ideas and
contributions to the development of the LEOSat Python package:

- [Jeremy Tregloan-Reed](mailto:jeremy.tregloan-reed@uda.cl), Universidad de Atacama
- [Eduardo Unda-Sanzana](mailto:eduardo.unda@uamail.cl), Universidad de Antofagasta
- [Edgar Ortiz](mailto:ed.ortizm@gmail.com), Universidad de Antofagasta
- [Maria Isabel Romero Colmenares](mailto:maria.romero.21@alumnos.uda.cl), Universidad de Atacama
- [Sangeetha Nandakumar](mailto:an.sangeetha@gmail.coml), Universidad de Atacama

The project would not have been possible without the help of everyone
who contributed.

## Feedback, questions, comments?

LEOSatpy is under active development and help with the development of new functionalities 
and fixing bugs is very much appreciated. In case you would like to contribute, 
feel free to fork the [GitHub repository](https://github.com/CLEOsat-group/leosatpy) and to create a pull request.

If you encounter a bug or problem, please [submit a new issue on the
GitHub repository](https://github.com/CLEOsat-group/leosatpy/issues)
providing as much detail as possible (error message, operating system,
Python version, etc.).

If you have further feedback, questions or comments you can also send an
e-mail to [Jeremy Tregloan-Reed](mailto:jeremy.tregloan-reed@uda.cl), or
[Christian Adam](mailto:christian.adam84@gmail.com).

## Author

[Christian Adam](mailto:christian.adam84@gmail.com), Centro de Investigación,
Tecnología, Educación y Vinculación Astronómica (CITEVA), Universidad de
Antofagasta, Antofagasta, Chile
