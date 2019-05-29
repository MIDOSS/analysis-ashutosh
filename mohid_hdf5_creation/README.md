This analysis folder organizes scripts, data and products used to  of model output towards understanding Salish Sea circulation, influences and impacts.

The links below are to static renderings of the notebooks via
[nbviewer.jupyter.org](https://nbviewer.jupyter.org/).
Descriptions under the links below are from the first cell of the notebooks
(if that cell contains Markdown or raw text).

* ##[explore_pytables_h5py.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/src/default/mohid_hdf5_creation/.//explore_pytables_h5py.ipynb)  
    
    <h1>Exploration of use of pytables and h5py libraries for creating forcing files for MOHID<h1>  
        <ol>  
            <li>  
                <a href="#winds">Winds Structure</a>  
            </li>  
            <li>  
                <a href="#currents">Currents Structure</a>  
            </li>  
            <li>  
                <a href="#createwind">Wind Input file pytables</a>  
            </li>  
            <li>  
                <a href="#createcurrents">Create Current Input file pytables</a>  
            </li>  
            <li>  
                <a href="#windsh5py">Create Wind Input file h5py</a>  
            </li>  
            <li>  
                <a href="#currentsh5py">Create Current Input file h5py</a>  
            </li>  
            <li>  
                <a href="#comparison">Looking at file size and time incentive for different compression levels</a>  
            </li>  
        </ol>  

* ##[HDF5 File Creation and Conventions.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/src/default/mohid_hdf5_creation/.//HDF5 File Creation and Conventions.ipynb)  
    
    **HDF5 File Creation and Conventions Documentation**  
      
    **Based on my experience with using the __[h5py](http://docs.h5py.org/en/stable/)__ library to create forcing files for MOHID, this notebook documents the recommended way of creating HDF5 files with a tree structure, compression variables and metadata attributes for datasets.**  

* ##[Hdf5_to_nc.ipynb](https://nbviewer.jupyter.org/urls/bitbucket.org/midoss/analysis-ashutosh/src/default/mohid_hdf5_creation/.//Hdf5_to_nc.ipynb)  
    

##License

These notebooks and files are copyright 2013-2019
by the Salish Sea MEOPAR Project Contributors
and The University of British Columbia.

They are licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
Please see the LICENSE file for details of the license.
