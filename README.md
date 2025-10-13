# Code for "Glacier processes from seismic recordings on Sørsdal Glacier, East Antarctica"
Jared C. Magyar, 2025

### Background

The scripts in this repository can be used to reproeduce the results in "Glacier processes from seismic recordings on Sørsdal Glacier, East Antarctica", the pre-print of which can be found at (LINK). It also provides an example end-to-end workflow for using the 'cryoquake' analysis tools, currently under development, and can be found at github.com/JMagyar15/cryoquake.

### Instructions

The scripts and notebooks included here require installation of the 'cryoquake' workflow, available at (LINK). This is under development, but can be installed by cloning/downloading the respository and using ``pip install -e /path/to/cryoquake``. 

Before running any of the scripts, the Sørsdal data must be downloaded. This can be accessed using the Obspy MassDownloader, or using the ``download_waveforms.py`` script included here. 

The event detection catalogues are included in the repository, so any other script can be run without needing to re-do the event detection. However, the event detection can be re-run using ``full_catalogue.py``. NOTE: as response is removed in day-long blocks, the event detection is sensitive to missing data from ANY part of the day. We have found that the data can sometimes be dowloaded a little inconsistantly, so minute differences in the event timings (altering the event IDs) is possible when running locally. We therefore suggest using the pre-computed catalogues in the repository.

