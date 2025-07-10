import os
from setuptools import setup, find_packages

version = {}
with open('leosatpy/utils/version.py') as fp:
    exec(fp.read(), version)

with open("README.md", "r") as fh:
    long_description = fh.read()

# Install requires from requirements.txt
requirementPath = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='leosatpy',
    version=version['__version__'],
    author='Christian Adam',
    author_email='christian.adam84@gmail.com',
    license='GNU General Public License v3.0',
    license_files=[],
    description='LEOSatpy is a highly-automated end-to-end pipeline for the reduction, calibration, '
                'and analysis of Low Earth Orbit Satellite observations from various telescopes.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='http://leosatpy.readthedocs.io/',
    download_url='https://github.com/CLEOsat-group/leosatpy/archive/master.zip',
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe=False,
    python_requires='>=3.9',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    include_package_data=True,  # So that non .py files make it onto pypi, and then back !
    entry_points={
        'console_scripts': [
            'reduceSatObs=leosatpy.reduceSatObs:main',
            'calibrateSatObs=leosatpy.calibrateSatObs:main',
            'analyseSatObs=leosatpy.analyseSatObs:main'
        ]
    },
)
