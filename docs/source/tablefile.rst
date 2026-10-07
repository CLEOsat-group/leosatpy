Result table
============

LEOSatpy provides a table to store the results of the analysis performed on the satellite trails.
This table is used to keep track of the observations, their metadata, and the results of the analysis.
(:file:`results_leosatpy.csv`)

The table can be separated into four groups:
1. Input metadata: Information about the observation, instrument, and satellite.
2. TLE-based predictions: Information derived from the TLE data, such as satellite position and velocity.
3. Measured trail properties: Information about the detected satellite trail, such as length and brightness.
4. Quality indicators: Flags and metrics to assess the quality of the measurements and analysis.

- :ref:`Observation information <obs_info>`
- :ref:`TLE-based predictions <tle_predictions>`
- :ref:`Trail properties <trail_properties>`
- :ref:`Additional information <additional_info>`

.. _obs_info:

Observation information
-----------------------

.. py:function:: File:

    **Type**: ``str``

    Source file name.

.. py:function:: Object:

   Object name.

.. py:function:: Sat-Name:

   Satellite common name (from TLE).

.. py:function:: AltID:

   Alternate identifier for the satellite or target.

.. py:function:: UniqueID:

   Unique record identifier from the `track.py`.

.. py:function:: Instrument:

   Instrument/camera identifier.

.. py:function:: HDU_idx:

   FITS HDU (Header/Data Unit) index.

.. py:function:: DetPosID:

   Detector position/segment identifier.

.. py:function:: Telescope:

   Telescope identifier.

.. py:function:: RA:

    **Type**: ``str``
    **Format**: ``hh:mm:ss.sss``

    Right ascension of field center (hh:mm:ss.sss).

.. py:function:: DEC:

   ``deg:mm:ss.sss``

   Declination of field center (deg:mm:ss.sss).

.. py:function:: Date-Obs:

   Observation date in UTC.

.. py:function:: Filter:

   Photometric filter/band.

.. py:function:: ExpTime:

   **Type**: ``float``
   **Unit**: ``s``

   Exposure time.

.. py:function:: Airmass:

   Airmass at Obs-Start.

.. py:function:: Binning:

   CCD binning factor.

.. py:function:: Obs-Start:

   ``hh:mm:ss.sss``

   Observation start in UTC.

.. py:function:: Obs-Mid:

   ``hh:mm:ss.sss``

   Observation mid-point in UTC.

.. py:function:: Obs-Stop:

   ``hh:mm:ss.sss``

   Observation end in UTC.

.. py:function:: HasTrail:

   Indicates if the image has a trail detected.

.. py:function:: NTrail:

   Number of detected trails.

.. py:function:: HasRef:

   ``bool``

   Indicates if at least one image without trail is avaialble.

.. py:function:: NRef:

   Number of available reference images.

.. _tle_predictions:

TLE-based predictions
----------------------

.. py:function:: UT Date:

   ``YYYY-MM-DD``

   Observation date in UTC from TLE.

.. py:function:: UT time:

   ``hh:mm:ss.sss``

   Observation time from TLE (hh:mm:ss.sss).

.. py:function:: SatLon:

   ``deg``

   Satellite longitude from TLE.

.. py:function:: SatLat:

   ``deg``

   Satellite latitude from TLE.

.. py:function:: SatAlt:

   ``km``

   Satellite altitude above ground from TLE.

.. py:function:: SatAz:

   ``deg``

   Satellite azimuth.

.. py:function:: SatElev:

   ``deg``

   Satellite elevation.

.. py:function:: SatRA:

   ``hh:mm:ss.sss``

   Satellite right ascension (hh:mm:ss.sss).

.. py:function:: SatDEC:

   ``deg``

   Satellite declination.

.. py:function:: SunRA:

   ``hr``

   Sun right ascension (hr).

.. py:function:: SunDEC:

   ``deg``

   Sun declination.

.. py:function:: SunZenithAngle:

   ``deg``

   Sun zenith angle.

.. py:function:: SatAngularSpeed:

   ``arcsec/s``

   Satellite angular velocity from TLE (arcsec/s).

.. py:function:: e_SatAngularSpeed:

   ``arcsec/s``

   1σ uncertainty of angular velocity (arcsec/s).

.. py:function:: ObsSatRange:

   ``km``

   Distance between satellite and observer (km).

.. py:function:: ToD:

   ``s``

   Time the satellite was on the detector.

.. py:function:: e_ToD:

   ``s``

    1σ uncertainty of ToD.

.. _trail_properties:

Trail properties
----------------

.. py:function:: ObsTrailLength:

   ``arcsec``

   Measured length of the trail in the image.

.. py:function:: e_ObsTrailLength:

   ``arcsec``

   1σ uncertainty of ObsTrailLength.

.. py:function:: EstTrailLength:

   ``arcsec``

   Estimated trail length using ExpTime and SatAngularSpeed.

.. py:function:: e_EstTrailLength:

   ``arcsec``

   1σ uncertainty of EstTrailLength.

.. py:function:: SunSatAng:

   ``deg``

   Angle between the sun and the satellite.

.. py:function:: SunPhaseAng:

   ``deg``

   Solar phase angle. Combines ObsAng and SunIncAng.

.. py:function:: SunIncAng:

   ``deg``

   Solar incedence angle (90 - SunElevAng) measured from the satellite normal.

.. py:function:: ObsAng:

   ``deg``

   Angle between satellite and observer measured from the satellite normal.

.. py:function:: ObsMag:

   ``mag``

   Observed trail magnitude.

.. py:function:: e_ObsMag:

   ``mag``

   1σ uncertainty of ObsMag.

.. py:function:: EstMag:

   ``mag``

   Observed magnitude corrected for ToD.

.. py:function:: e_EstMag:

   ``mag``

   1σ uncertainty of EstMag.

.. py:function:: EstScaleMag:

   ``mag``

   Estimated magnitude scaled to the satellite referecne altitude.

.. py:function:: e_EstScaleMag:

   ``mag``

   1σ uncertainty of EstScaleMag.

.. py:function:: HasZp:

   ``bool``

   Indicates whether the observation has absolute photometric calibration.

.. py:function:: MagZp:

   ``mag``

   Magnitude zero point.

.. py:function:: ObsMag_zp:

   ``mag``

   Calibrated observed trail magnitude.

.. py:function:: e_ObsMag_zp:

   ``mag``

   1σ uncertainty of ObsMag_zp.

.. py:function:: EstMag_zp:

   ``mag``

   Calibrated observed magnitude corrected for ToD.

.. py:function:: e_EstMag_zp:

   ``mag``

   1σ uncertainty of EstMag_zp.

.. py:function:: EstScaleMag_zp:

   ``mag``

   Calibrated estimated magnitude scaled to a ref. height.

.. py:function:: e_EstScaleMag_zp:

   ``mag``

   1σ uncertainty of EstScaleMag_zp.

.. py:function:: SunAzAng:

   ``deg``

   Sun azimuth angle.

.. py:function:: SunElevAng:

   ``deg``

   Sun elevation angle.


.. py:function:: FluxScale:

   Scale factor used to correct the observed magnitude.

.. py:function:: MagScale:

   Scale factor used to normalize the photometry to a nominal orbit height.

.. py:function:: MagCorrect:

   ``mag``

   Estimated magnitude correction for referecne stars.

.. py:function:: e_MagCorrect:

   ``mag``

   1σ uncertainty of MagCorrect.

.. py:function:: MagCorrectSat:

   ``mag``

   Estimated magnitude correction for satellite.

.. py:function:: dt_tle-obs:

   ``s``

   Difference between predicted and actual observation time.

.. py:function:: TrailCX:

   ``px``

   Trail aperture center in x-direction.

.. py:function:: e_TrailCX:

   ``px``

   1σ uncertainty of TrailCX.

.. py:function:: TrailCY:

   ``px``

   Trail aperture center in y-direction.

.. py:function:: e_TrailCY:

   ``px``

   1σ uncertainty of TrailCY.

.. py:function:: TrailCRA:

   ``deg``

   Trail center expressed in right ascension.

.. py:function:: e_TrailCRA:

   ``arcsec``

   1σ uncertainty of TrailCRA.

.. py:function:: TrailCDEC:

   ``deg``

   Trail center expressed in declination.

.. py:function:: e_TrailCDEC:

   ``arcsec``

   1σ uncertainty of TrailCDEC.

.. py:function:: TrailANG:

   ``deg``

   Trail aperture orientation.

.. py:function:: e_TrailANG:

   ``arcsec``

   1σ uncertainty of TrailANG.

.. py:function:: OptAperHeight:

   ``px``

   Trail aperture height.


.. _additional_info:

Additional information
-----------------------

.. py:function:: CalRA:

   ``deg``

   WCS-calibrated right ascension of measurement position.

.. py:function:: CalDEC:

   ``deg``

   WCS-calibrated declination of measurement position.

.. py:function:: CentX:

   ``px``

   Centroid x-coordinate.

.. py:function:: e_CentX:

   ``px``

   1σ uncertainty of CentX.

.. py:function:: CentY:

   ``px``

   Centroid y-coordinate.

.. py:function:: e_CentY:

   ``px``

   1σ uncertainty of CentY.

.. py:function:: PixScale:

   ``arcsec/px``

   Plate scale.

.. py:function:: DetRotAng:

   ``deg``

   Detector rotation angle on sky.


.. py:function:: FWHM:

   ``px``

   Stellar full width at half maximum.

.. py:function:: e_FWHM:

   ``px``

   1σ uncertainty of FWHM.

.. py:function:: bias_cor:

   ``bool``

   Flag indicating bias correction applied.

.. py:function:: dark_cor:

   ``bool``

   Flag indicating dark correction applied.

.. py:function:: flat_cor:

   ``bool``

   Flag indicating flat correction applied.

.. py:function:: WCS_cal:

   ``bool``

   Astrometric WCS calibration status.

.. py:function:: mag_conv:

   ``bool``

   Photometric conversion due to missing photometry in the observed filter.

.. py:function:: QlfAperRad:

   ``bool``

   Flag indicating optimum aperture radius determination applied.

.. py:function:: TrailDetMethod:

   ``str``

   Method used to detect the satellite trail. Auto/Manual