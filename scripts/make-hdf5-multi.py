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

import os
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np
import xarray as xr
import errno
import time
import h5py
from salishsea_tools import utilities
from salishsea_tools import viz_tools
import multiprocessing
from mpi4py import MPI
import mpi4py
#from scipy.interpolate import griddata

# NEMO input files directory
nemoinput = '/results2/SalishSea/nowcast-green.201806/'

# HRDPS input files directory
hdinput = '/results/forcing/atmospheric/GEM2.5/operational/'

# WW3 input files directory
wwinput = '/opp/wwatch3/nowcast/'

# Output filepath
outpath = '/results2/MIDOSS/forcing/SalishSeaCast/'


## Integer -> String
## consumes time in seconds and outputs a string that gives the time in HH:MM:SS format
def conv_time(time):
    """Give time in HH:MM:SS format.

    :arg time: time elapsed in seconds
    :type integer: :py:class:'int'

    :returns: time elapsed in HH:MM:SS format
    :rtype: :py:class:`str'
    """
    hours = int(time/3600)
    mins = int((time - (hours*3600))/60)
    secs = int((time - (3600 * hours) - (mins *60)))
    return '{}:{}:{}'.format(hours, mins, secs)


def generate_currents_hdf5(timestart, timeend, path, outpath, compression_level = 1):
    """Provide paths, groups and parameters for multiprocessing

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

    :returns f: <HDF5 file (mode r+)>
    :rtype: :py:class:`File'
    
    :returns U_files: listofString: U parameter file paths
    :rtype: :py:class:`list'
    
    :returns V_files: listofString: V parameter file paths
    :rtype: :py:class:`list'
    
    :returns T_files: listofString: T parameter file paths
    :rtype: :py:class:`list'
    
    :returns times: HDF5 group for time
    :rtype: :py:class:`Group'
    
    :returns velocity_u: HDF5 group for U velocities
    :rtype: :py:class:`Group'
    
    :returns velocity_v: HDF5 group for V velocities
    :rtype: :py:class:`Group'
    
    :returns water_level: HDF5 group for sea surface heights
    :rtype: :py:class:`Group'

    :returns compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'
    """
    
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]
    U_files = []
    V_files = []
    T_files = []

    # append all filename strings within daterange to lists
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d')
        
        # check if file exists. exit if it does not. add path to list if it does.
            # U files
        U_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_U.nc'
        if not os.path.exists(U_path):
            print(f'File {U_path} not found. Check Directory and/or Date Range.')
            return
        U_files.append(U_path)
            # V files
        V_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_V.nc'
        if not os.path.exists(V_path):
            print(f'File {V_path} not found. Check Directory and/or Date Range.')
            return
        V_files.append(V_path)
            # T files
        T_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_T.nc'
        if not os.path.exists(T_path):
            print(f'File {T_path} not found. Check Directory and/or Date Range.')
            return
        T_files.append(T_path)
        
    print('\nAll source files found')
    # string: output folder name with date ranges used. end date will be lower by a day than timeend because datasets only go until midnight
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    # create output directory
    dirname = f'{outpath}hdf5/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    print(f'\nOutput directory {dirname} created\n')
    # create hdf5 file and create tree structure

    return U_files, V_files, T_files, dirname, compression_level
    
#for file_index in bar(range(number_of_files)):
def write_currents (U_files, V_files, T_files, compression_level, file_index, times, water_level):
    """"   
    :arg U_files: listofString: U parameter file paths
    :type: :py:class:`list'
    
    :arg V_files: listofString: V parameter file paths
    :type: :py:class:`list'
    
    :arg T_files: listofString: T parameter file paths
    :type: :py:class:`list'
    
    :arg times: HDF5 group for time
    :type: :py:class:`Group'
    
    :arg velocity_u: HDF5 group for U velocities
    :type: :py:class:`Group'
    
    :arg velocity_v: HDF5 group for V velocities
    :type: :py:class:`Group'
    
    :arg water_level: HDF5 group for sea surface heights
    :type: :py:class:`Group'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'
    """
    

    T_raw = xr.open_dataset(T_files[file_index])
    # assume all files have same time_counter markers
    datelist = T_raw.time_counter.values.astype('datetime64[s]').astype(datetime)
    # unstagger to move U, V to center of grid square

    # convert xarrays to numpy arrays and cut off grid edges

    sea_surface = T_raw.sossheig.values[...,:,1:897:,1:397]

    sea_surface = np.transpose(sea_surface, [0,2,1])

    sea_surface = np.nan_to_num(sea_surface).astype('float64')
    attr_counter = file_index * sea_surface.shape[0]
    # make list of time arrays
    datearrays = []
    for date in datelist:
        datearrays.append(np.array([date.year, date.month, date.day, date.hour, date.minute, date.second]).astype('float64'))


    for i in range(sea_surface.shape[0]):
        level_attr = 'water level_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
        dset = water_level.create_dataset(level_attr, shape = (396, 896), data = sea_surface[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
        metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([5.]), 'Minimum' : np.array([-5.]), 'Units' : b'm'}
        dset.attrs.update(metadata)

    # write time values to hdf5

    for i in range(len(datearrays)):
        print(i + attr_counter + 1)
        time_attr = 'Time_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
        dset = times.create_dataset(time_attr, shape = (6,), data = datearrays[i],chunks=(6,), compression = 'gzip', compression_opts = compression_level)
        metadata = {'Maximum' : np.array([datearrays[i][0].astype('float64')]), 'Minimum' : np.array([-0.]), 'Units' : b'YYYY/MM/DD HH:MM:SS'} # !!!
        dset.attrs.update(metadata)
    return
    
def manage_queue(current, remaining, workers):
    if len(current) != 0:
        for task in current:
            if task.is_alive():
                continue
            else:
                try:
                    task.join()
                    current.remove(task)
                except RuntimeError as err:
                    if 'cannot join current thread' in err.args[0]:
                        continue
                    else:
                        raise
    if len(current) != workers and len(remaining) != 0:
        remaining[0].start()
        current.append(remaining[0])
        remaining.remove(remaining[0])
        time.sleep(1)
        manage_queue(current, remaining, workers)
    if len(current) == 0  and  len(remaining) == 0:
        return
    else:
        time.sleep(1)
        manage_queue(current, remaining, workers)

                
if __name__ == '__main__':
    beganat = time.time()
    timestart = '1 Feb 2019'
    timeend = '5 Feb 2019'
    U_files, V_files, T_files, dirname,compression_level = generate_currents_hdf5(timestart, timeend, nemoinput, outpath, compression_level = 1)
    f = h5py.File(f'temp/foocurrents.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
    times = f.create_group('Time')
    water_level = f.create_group('/Results/water level')
    processes= []
    #multiprocessing.set_start_method('spawn')
    for i in range(len(U_files)):
        p = multiprocessing.Process(target = write_currents, args = (U_files, V_files, T_files, compression_level, i, times, water_level))
        processes.append(p)
    manage_queue([], processes, 4)
    f.close()
    print('Time elapsed: {}'.format(conv_time(time.time()-beganat)))