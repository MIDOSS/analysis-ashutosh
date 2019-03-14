# Copyright 2013-2016 The Salish Sea MEOPAR contributors
# and The University of British Columbia

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates forcing HDF5 input files for MOHID based on user input time range
"""

import errno
import h5py
import numpy as np
import os
import time
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse
from salishsea_tools import utilities
from salishsea_tools import viz_tools

# NEMO input files directory
nemoinput = '/results2/SalishSea/nowcast-green.201806/'

def timer(func):
    """Decorator function for timing function calls
    """
    def f(*args, **kwargs):
        beganat = time.time()
        rv = func(*args, *kwargs)
        elapsed = time.time() - beganat
        hours = int(elapsed / 3600)
        mins = int((elapsed - (hours*3600))/60)
        secs = int((elapsed - (hours*3600) - (mins*60)))
        print('\nTime elapsed: {}:{}:{}\n'.format(hours, mins, secs))
        return rv
    return f

def e3t_paths(timestart, timeend, path, outpath, compression_level = 1):
    """Generate paths for Salish Seacast e3t files

    :arg timestart: date from when to start concatenating
    :type string: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type string: :py:class:'str'

    :arg path: path of input files
    :type string: :py:class:'str'

    :arg outpath: path for output files
    :type string: :py:class:'str'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'

    :returns tuple: tuple containing the arguments to pass to hdf5 file generator functions
    :rtype: :py:class:`tuple'
    """
    
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]

    # append all filename strings within daterange to lists
    e3t_files = []
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days = day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d')
        
        # check if file exists. exit if it does not. add path to list if it does.

            # e3t files
        e3t_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_carp_T.nc'
        if not os.path.exists(e3t_path):
            print(f'File {e3t_path} not found. Check Directory and/or Date Range.')
            return False
        e3t_files.append(e3t_path)
        print('\nAll source files found')
    
    # string: output folder name with date ranges used. 
    # end date will be lower by a day than timeend because datasets only go until midnight
    startfolder, endfolder = parse(timestart), parse(timeend) - timedelta(1)
    folder = str(
        datetime(startfolder.year, startfolder.month,startfolder.day).strftime('%d%b%y').lower()
        ) + '-' + str(
            datetime(endfolder.year, endfolder.month, endfolder.day).strftime('%d%b%y').lower()
            )

    # create output directory
    dirname = f'{outpath}MF0/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print(exc.errno)
                return False
    
    print(f'\nSalish SeaCast output directory {dirname} created\n')
    
    return (e3t_files, dirname, compression_level)

def create_e3t_hdf5(e3t_files, dirname, compression_level = 1):
    """Generate e3t files for MOHID

    :arg W_files: listofString; Salish SeaCast e3t netcdf file paths
    :type list: :py:class:'list'

    :arg dirname: Output file directory
    :type string: :py:class:'str'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # create hdf5 file and create tree structure
    f = h5py.File(f'{dirname}e3t.hdf5', 'w')
    times = f.create_group('Time')
    e3t = f.create_group('/Results/e3t')

    # since we are looping through the source files by day, we want to keep track of the
    # number of records we have made so that we can allocate the correct child names
    attr_counter = 0

    number_of_files = len(e3t_files)
    bar = utilities.statusbar('Creating e3t file ...')
    for file_index in bar(range(number_of_files)):
        # load NEMO netcdf source files using xarray
        e3t_raw = xr.open_dataset(e3t_files[file_index])

        # load dates from e3t netcdf file
        datelist = e3t_raw.time_counter.values.astype('datetime64[s]').astype(datetime)

        # convert xarray DataArrays to numpy arrays and cut off grid edges
        e3t = e3t_raw.e3t.values[...,:,1:897:,1:397]
        
        # clear memory
        del(e3t_raw)
        
        # transpose grid (rotate 90 clockwise)
        e3t = np.transpose(e3t, [0,1,3,2])

        # flip currents by depth dimension
        e3t = np.flip(e3t, axis = 1)

        # convert nans to 0s and set datatype to float64
        e3t = np.nan_to_num(e3t).astype('float64')
        
        # make list of time arrays
        datearrays = []
        for date in datelist:
            datearrays.append(
                np.array([date.year,date.month, date.day, date.hour, date.minute, date.second]).astype('float64')
                )

        # write time values to hdf5
        for i, datearray in enumerate(datearrays):
            child_name = 'Time_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = times.create_dataset(
                child_name,
                shape = (6,),
                data = datearray,
                chunks = (6,),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'Maximum' : np.array(datearrays[i][0]),
                'Minimum' : np.array([-0.]),
                'Units' : b'YYYY/MM/DD HH:MM:SS'
                } 
            dset.attrs.update(metadata)

        # write u current values to hdf5
        for i, dataset in enumerate(e3t):
            child_name = 'e3t_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = e3t.create_dataset(
                child_name,
                shape = (40, 396, 896),
                data = dataset,
                chunks = (40, 396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Units' : b'm'
                }
            dset.attrs.update(metadata)
        
        # update the accumulator
        attr_counter = attr_counter + e3t.shape[0]
        
        # clear memory
        del(e3t)

    f.close()
    return
