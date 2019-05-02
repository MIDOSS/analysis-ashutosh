This analysis folder organizes scripts, data and products used to  of model output towards understanding Salish Sea circulation, influences and impacts.

The links below are to static renderings of the notebooks via
[nbviewer.jupyter.org](https://nbviewer.jupyter.org/).
Descriptions under the links below are from the first cell of the notebooks
(if that cell contains Markdown or raw text).

* ##[River_plume.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//River_plume.ipynb)  
    
    **Freshet analysis**  

* ##[Make_profile.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Make_profile.ipynb)  
    
    **This notebook provides a recipe for creating netcdf files of profiles by looping through the files on in /results2, /results, /opp**  
    - It has been heavily generalised to provide a general recipe for making the profiles, rather than providing explicit functions that do the job.  
    - I would estimate that it would take about an hour to concatenate a year's worth of data. Consider a multiprocessing.py approach to creating multiple profiles on a queue.   

* ##[mohid_viz.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//mohid_viz.ipynb)  
    
    **MOHID visualisation tools**  

* ##[Stratification.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Stratification.ipynb)  
    
    **Stratification vs Depth Plots**  

* ##[SpringNeapTide.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//SpringNeapTide.ipynb)  
    
* ##[Salinity_Regression.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Salinity_Regression.ipynb)  
    
    **Discharge Vs Salinity in SoG**  

* ##[current_analysis.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//current_analysis.ipynb)  
    
    **Compare the 1 hour interval currents on the full SSC grid vs the 20 mins interval currents over a part of the domain **  
    - Apply a low pass and band pass filter on the 20 mins current speeds at the three chosen locations to look at the fornightly cycles  
    - Apply a running average on the 20 min interval currents to reduce the noise in the signal  

* ##[Buoyancy_correlation.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Buoyancy_correlation.ipynb)  
    
    **Buoyancy Correlation**  
    **Use a linear regression to determine and plot the correlation between Fraser river discharge at Hope with the Salinity at the SoG point**  

* ##[Drifter Locations.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Drifter Locations.ipynb)  
    
    **Look at Rich's Drifter Relase Locations on Nemo Grid. Find the release locations that are closest to our three chosen points (by a given radius).**  

* ##[Pick grid points.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Pick grid points.ipynb)  
    
* ##[wind_climatology_maps.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//wind_climatology_maps.ipynb)  
    
    **Wind Climatology Maps**  

* ##[WindSpeedTimeseries.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//WindSpeedTimeseries.ipynb)  
    
    **Wind Speed Time Series**  

* ##[Compare_ssh_currents_winds.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/raw/default/climatology_analysis_notebooks/.//Compare_ssh_currents_winds.ipynb)  
    
    **Produce plots comparing the spring/neap cycle wityh the timeseries of winds, surface current speeds and surface salinity at the three chosen locations. **  
    **Includes functionality for running means, and the ability to zoom into a time window to get a more detaled look at what is happining within a chosen time window.**  


##License

These notebooks and files are copyright 2013-2019
by the Salish Sea MEOPAR Project Contributors
and The University of British Columbia.

They are licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
Please see the LICENSE file for details of the license.
