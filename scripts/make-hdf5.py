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
from scipy.interpolate import griddata

# NEMO input files directory
nemoinput = '/results2/SalishSea/nowcast-green.201806/'

# HRDPS input files directory
hdinput = '/results/forcing/atmospheric/GEM2.5/operational/'

# WW3 input files directory
wwinput = '/opp/wwatch3/nowcast/'

# Output filepath
outpath = '/results2/MIDOSS/forcing/SalishSeaCast/'

mask = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSn2DMeshMaskV17-02').tmaskutil.isel(time = 0).values[1:897, 1:397].T

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

def salishseacast_paths(timestart, timeend, path, outpath, compression_level = 1, mask = mask):
    """Generate paths for Salish Seacast forcing 

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

    :returns tuple: three tuples containing the arguments to pass to hdf5 file generator functions
    :rtype: :py:class:`tuple'
    """
    
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]

    # append all filename strings within daterange to lists
    U_files, V_files, W_files, T_files, e3t_files = [], [], [], [], []
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days = day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d')
        
        # check if file exists. exit if it does not. add path to list if it does.

            # U files
        U_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_U.nc'
        if not os.path.exists(U_path):
            print(f'File {U_path} not found. Check Directory and/or Date Range.')
            return False
        U_files.append(U_path)

            # V files
        V_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_V.nc'
        if not os.path.exists(V_path):
            print(f'File {V_path} not found. Check Directory and/or Date Range.')
            return False
        V_files.append(V_path)

            # W files
        W_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_W.nc'
        if not os.path.exists(W_path):
            print(f'File {W_path} not found. Check Directory and/or Date Range.')
            return False
        W_files.append(W_path)

            # T files
        T_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_T.nc'
        if not os.path.exists(T_path):
            print(f'File {T_path} not found. Check Directory and/or Date Range.')
            return False
        T_files.append(T_path)
        
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
    return (
        (U_files, V_files, W_files, dirname, compression_level),
        (T_files, dirname, compression_level),
        (e3t_files, dirname, compression_level, mask),
        )

def hrdps_paths(timestart, timeend, path, outpath, compression_level = 1):
    """Generate wind input file paths

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

    :returns tuple: tuple containing the arguments to pass to hdf5 file generator function
    :rtype: :py:class:`tuple'
    """
    
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]

    # append all filename strings within daterange to list
    wind_files = []
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        
        month = datestamp.month
        if month < 10:
            month = f'0{str(month)}'
        
        day = datestamp.day
        if day < 10:
            day = f'0{str(day)}'
        
        year = str(datestamp.year)

        # check if file exists. exit if it does not. add path to list if it does.
        wind_path = f'{path}ops_y{year}m{month}d{day}.nc'
        if not os.path.exists(wind_path):
            print(f'File {wind_path} not found. Check Directory and/or Date Range.')
            return 
        wind_files.append(wind_path)

    print('\nAll source files found')

    # string: output folder name with date ranges used. 
    # end date will be lower by a day than timeend because datasets only go until midnight
    startfolder, endfolder = parse(timestart), parse(timeend) - timedelta(1)
    folder = str(
        datetime(startfolder.year, startfolder.month, startfolder.day).strftime('%d%b%y').lower()
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
    
    print(f'\nHRDPS output directory {dirname} created\n')
    return (wind_files, dirname, compression_level)

def ww3_paths(timestart, timeend, path, outpath, compression_level = 1):
    """Generate Wave Watch 3 input files paths

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

    :returns tuple: tuple containing the arguments to pass to hdf5 file generator function
    :rtype: :py:class:`tuple'
    """
    # generate list of dates from daterange given
    months = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5 : 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9 : 'sep', 10: 'oct', 11 :'nov',12: 'dec'}
    daterange = [parse(t) for t in [timestart, timeend]]


    # append all filename strings within daterange to list
    wave_files = []
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        datestr2 = datestamp.strftime('%Y%m%d').lower()
        monthnm = months[datestamp.month]
        
        day = datestamp.day
        if day < 10:
            day = f'0{str(day)}'
        
        year = str(datestamp.year)[2:4]

        wave_path = f'{path}{day}{monthnm}{year}/SoG_ww3_fields_{datestr2}_{datestr2}.nc'
        if not os.path.exists(wave_path):
            print(f'File {wave_path} not found. Check Directory and/or Date Range.')
            return False
        wave_files.append(wave_path)

    print('\nAll source files found')
 
    # string: output folder name with date ranges used. 
    # end date will be lower by a day than timeend because datasets only go until midnight
    startfolder, endfolder = parse(timestart), parse(timeend) - timedelta(1)
    folder = str(
        datetime(startfolder.year, startfolder.month, startfolder.day).strftime('%d%b%y').lower()
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

    print(f'\nWW3 output directory {dirname} created\n')
    return (wave_files, dirname, compression_level)

@timer  
def create_currents_hdf5(U_files, V_files, W_files, dirname, compression_level = 1):
    """Generate currents forcing file for MOHID

    :arg U_files: listofString; Salish SeaCast U netcdf file paths
    :type list: :py:class:'list'

    :arg V_files: listofString; Salish SeaCast V netcdf file paths
    :type list: :py:class:'list'

    :arg W_files: listofString; Salish SeaCast W netcdf file paths
    :type list: :py:class:'list'

    :arg dirname: Output file directory
    :type string: :py:class:'str'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # create hdf5 file and create tree structure
    f = h5py.File(f'{dirname}currents.hdf5', 'w')
    times = f.create_group('Time')
    velocity_u = f.create_group('/Results/velocity U')
    velocity_v = f.create_group('/Results/velocity V')
    velocity_w = f.create_group('/Results/velocity W')

    # since we are looping through the source files by day, we want to keep track of the
    # number of records we have made so that we can allocate the correct child names
    attr_counter = 0

    number_of_files = len(U_files)
    bar = utilities.statusbar('Creating currents forcing file ...')
    for file_index in bar(range(number_of_files)):
        # load NEMO netcdf source files using xarray
        U_raw = xr.open_dataset(U_files[file_index])
        V_raw = xr.open_dataset(V_files[file_index])
        W_raw = xr.open_dataset(W_files[file_index])

        # load dates from U netcdf file
        datelist = U_raw.time_counter.values.astype('datetime64[s]').astype(datetime)

        # unstagger to move U, V to center of grid square
        U  = viz_tools.unstagger_xarray(U_raw.vozocrtx, 'x')
        V  = viz_tools.unstagger_xarray(V_raw.vomecrty, 'y')
        W  = W_raw.vovecrtz #!!! 

        # convert xarray DataArrays to numpy arrays and cut off grid edges
        current_u = U.values[...,:,1:897:,1:397]
        current_v = V.values[...,:,1:897:,1:397]
        current_w = W.values[...,:,1:897:,1:397]

        # rotate currents to True North
        #current_u, current_v = viz_tools.rotate_vel(U, V)
        # clear memory
        del(U_raw); del(V_raw); del(W_raw); del(U); del(V)
        
        # transpose grid (rotate 90 clockwise)
        current_u = np.transpose(current_u, [0,1,3,2])
        current_v = np.transpose(current_v, [0,1,3,2])
        current_w = np.transpose(current_w, [0,1,3,2])

        # flip currents by depth dimension
        current_u = np.flip(current_u, axis = 1)
        current_v = np.flip(current_v, axis = 1)
        current_w = np.flip(current_w, axis = 1)

        # convert nans to 0s and set datatype to float64
        current_u = np.nan_to_num(current_u).astype('float64')
        current_v = np.nan_to_num(current_v).astype('float64')
        current_w = np.nan_to_num(current_w).astype('float64')
        
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
        for i, dataset in enumerate(current_u):
            child_name = 'velocity U_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = velocity_u.create_dataset(
                child_name,
                shape = (40, 396, 896),
                data = dataset,
                chunks = (40, 396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([5.]),
                'Minimum' : np.array([-5.]),
                'Units' : b'm/s'
                }
            dset.attrs.update(metadata)
    
        # write v current values to hdf5
        for i, dataset in enumerate(current_v):
            child_name = 'velocity V_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = velocity_v.create_dataset(
                child_name,
                shape = (40, 396, 896),
                data = dataset,
                chunks = (40, 396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([5.]),
                'Minimum' : np.array([-5.]),
                'Units' : b'm/s'
                }
            dset.attrs.update(metadata)

        # write w current values to hdf5
        for i, dataset in enumerate(current_w):
            child_name = 'velocity W_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = velocity_w.create_dataset(
                child_name,
                shape = (40, 396, 896),
                data = dataset,
                chunks = (40, 396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([5.]),
                'Minimum' : np.array([-5.]),
                'Units' : b'm/s'
                }
            dset.attrs.update(metadata)
        
        # update the accumulator
        attr_counter = attr_counter + current_u.shape[0]
        
        # clear memory
        del(current_u); del(current_v); del(current_w)

    f.close()
    return

@timer
def create_t_hdf5(T_files, dirname, compression_level = 1):
    """Generate Wave Watch 3 input files

    :arg T_files: listofString; Salish SeaCast T netcdf file paths
    :type list: :py:class:'list'

    :arg dirname: Output file directory
    :type string: :py:class:'str'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'
    
    :returns: None
    :rtype: :py:class:`NoneType'
    """

    # create hdf5 file and tree structure
    f = h5py.File(f'{dirname}t.hdf5', 'w')
    times = f.create_group('Time')
    water_level = f.create_group('/Results/water level')
    vosaline = f.create_group('/Results/salinity')
    votemper = f.create_group('/Results/temperature')

    # since we are looping through the source files by day, we want to keep track of the
    # number of records we have made so that we can allocate the correct child names
    attr_counter = 0

    bar = utilities.statusbar('Creating T parameters file ...', maxval = len(T_files))
    for t_file in bar(T_files):
        # load NEMO netcdf source files using xarray
        T_raw = xr.open_dataset(t_file)

        # convert xarray DataArrays to numpy arrays and cut off grid edges
        salinity = T_raw.vosaline.values[...,:,1:897:,1:397]
        temperature = T_raw.votemper.values[...,:,1:897:,1:397]
        sea_surface = T_raw.sossheig.values[...,:,1:897:,1:397]

        # load dates from T netcdf file
        datelist = T_raw.time_counter.values.astype('datetime64[s]').astype(datetime)

        del(T_raw)

        # transpose grid (rotate 90 clockwise)
        salinity = np.transpose(salinity, [0,1,3,2])
        temperature = np.transpose(temperature, [0,1,3,2])
        sea_surface = np.transpose(sea_surface, [0,2,1])

        # flip by depth dimension
        salinity = np.flip(salinity, axis = 1)
        temperature = np.flip(temperature, axis = 1)

        # convert nans to 0s and set datatype to float64
        salinity = np.nan_to_num(salinity).astype('float64')
        temperature = np.nan_to_num(temperature).astype('float64')
        sea_surface = np.nan_to_num(sea_surface).astype('float64')
        
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
        
        # write salinity values to hdf5
        for i, dataset in enumerate(salinity):
            child_name = 'salinity_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = vosaline.create_dataset(
                child_name,
                shape = (40, 396, 896),
                data = dataset,
                chunks = (40, 396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([100.]),
                'Minimum' : np.array([-100.]),
                'Units' : b'psu'
                }
            dset.attrs.update(metadata)

        # write temperature values to hdf5
        for i, dataset in enumerate(temperature):
            child_name = 'temperature_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = votemper.create_dataset(
                child_name,
                shape = (40, 396, 896),
                data = dataset,
                chunks = (40, 396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([100.]),
                'Minimum' : np.array([-100.]),
                'Units' : b'?C'
                }
            dset.attrs.update(metadata)
    
        # write  water level values to hdf5
        for i, dataset in enumerate(sea_surface):
            child_name = 'water level_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = water_level.create_dataset(
                child_name,
                shape = (396, 896),
                data = dataset,
                chunks = (396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([5.]),
                'Minimum' : np.array([-5.]),
                'Units' : b'm'
                }
            dset.attrs.update(metadata)
        
        # update the accumulator
        attr_counter = attr_counter + salinity.shape[0]
        
        # clear memory
        del(salinity); del(temperature); del(sea_surface)

    f.close()
    return

@timer
def create_winds_hdf5(wind_files, dirname, compression_level = 1):
    """Generate wind forcing input files

    :arg wind_files: listofString; HRDPS netcdf file paths
    :type list: :py:class:'list'

    :arg dirname: Output file directory
    :type string: :py:class:'str'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    
    # create hdf5 file and create tree structure
    f = h5py.File(f'{dirname}winds.hdf5', 'w')
    times = f.create_group('Time')
    windu = f.create_group('/Results/wind velocity X')
    windv = f.create_group('/Results/wind velocity Y')
    
    # since we are looping through the source files by day, we want to keep track of the
    # number of records we have made so that we can allocate the correct child names    
    attr_counter = 0

    bar = utilities.statusbar('Creating winds forcing file ...', maxval = len(wind_files))
    for wind_file in bar(wind_files):
        # load HRDPS netcdf file using xarray
        GEM = xr.open_dataset(wind_file)

        # lat lon data
        GEM_grid = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSaAtmosphereGridV1')
        NEMO_grid = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSnBathymetryV17-02')
        
        # GEM data coordinates
        points = np.array([GEM_grid.latitude.values.ravel(), GEM_grid.longitude.values.ravel()-360]).T
        
        # NEMO lat lon grids tuple
        xi = (NEMO_grid.latitude.values, NEMO_grid.longitude.values)
        
        # GEM Data
        GEM_u = GEM.u_wind.values
        GEM_v = GEM.v_wind.values

        # load dates from netcdf file
        datelist = GEM.time_counter.values.astype('datetime64[s]').astype(datetime)
        
        del(GEM)

        # create an interpolated array of the 1st slice so you can stack it onto the rest
        u_wind = np.expand_dims(griddata(points, GEM_u[0].ravel(), xi, method='cubic'),0)
        v_wind = np.expand_dims(griddata(points, GEM_v[0].ravel(), xi, method='cubic'),0)
        
        # interpolate and stack the rest of the grids
        for grid in range(1, GEM_u.shape[0]):
            interp_u = griddata(points, GEM_u[grid].ravel(), xi, method='cubic')
            u_wind = np.vstack((u_wind, np.expand_dims(interp_u,0)))

            interp_v = griddata(points, GEM_v[grid].ravel(), xi, method='cubic')
            v_wind = np.vstack((v_wind, np.expand_dims(interp_v,0)))

        # cut off grid edges
        u_wind = u_wind[...,:,1:897:,1:397]
        v_wind = v_wind[...,:,1:897:,1:397]

        # transpose grid (rotate 90 clockwise)
        u_wind = np.transpose(u_wind, [0,2,1])
        v_wind = np.transpose(v_wind, [0,2,1])

        # set datatype to float64
        u_wind = u_wind.astype('float64')
        v_wind = v_wind.astype('float64')
        
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
                'Maximum' : np.array([float(datearrays[i][0])]),
                'Minimum' : np.array([-0.]),
                'Units' : b'YYYY/MM/DD HH:MM:SS'
                } 
            dset.attrs.update(metadata)

        # write X wind velocities to hdf5
        for i, dataset in enumerate(u_wind):
            child_name = 'wind velocity X_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = windu.create_dataset(
                child_name,
                shape  = (396, 896),
                data = dataset,
                chunks = (396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([100.]),
                'Minimum' : np.array([-100.]),
                'Units' : b'm/s'
                }
            dset.attrs.update(metadata)
        
        # write Y wind velocities to hdf5
        for i, dataset in enumerate(v_wind):
            child_name = 'wind velocity Y_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = windv.create_dataset(
                child_name,
                shape = (396, 896),
                data = dataset,
                chunks = (396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([100.]),
                'Minimum' : np.array([-100.]),
                'Units' : b'm/s'
                }
            dset.attrs.update(metadata)
        # update the accumulator 
        attr_counter = attr_counter + u_wind.shape[0]

        # clear memory
        del(u_wind); del(v_wind)
    f.close()
    return

@timer
def create_ww3_hdf5(wave_files, dirname, compression_level = 1):
    """Generate Wave Watch 3 input files

    :arg wave_files: listofString; WW3 netcdf file paths
    :type list: :py:class:'list'

    :arg dirname: Output file directory
    :type string: :py:class:'str'

    :arg compression_level: compression level for output file (Integer[1,9])
    :type integer: :py:class:'int'
    
    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # create hdf5 file and create tree structure
    f = h5py.File(f'{dirname}ww3.hdf5', 'w')
    times = f.create_group('Time')
    mwp = f.create_group('/Results/mean wave period')
    swh = f.create_group('/Results/significant wave height')
    wc  = f.create_group('/Results/whitecap coverage')

    # since we are looping through the source files by day, we want to keep track of the
    # number of records we have made so that we can allocate the correct child names
    attr_counter = 0

    bar = utilities.statusbar('Creating WW3 parameters file ...', maxval = len(wave_files))
    for wave_file in bar(wave_files):
        # load WW3 netcdf source file using xarray
        WW3 = xr.open_dataset(wave_file)

        # SalishSeaCast lat-lon data
        NEMO_grid = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSnBathymetryV17-02')
       
        
	# WW3 data coordinates
        lat_lon_mesh = np.meshgrid(WW3.MAPSTA.latitude.values, WW3.MAPSTA.longitude.values)    
        points = np.array([lat_lon_mesh[0].T.ravel(), lat_lon_mesh[1].T.ravel()-360]).T
       
        # NEMO lat lon grids tuple
        xi = (NEMO_grid.latitude.values, NEMO_grid.longitude.values)
       
        # convert xarray DataArrays to numpy arrays
        mean_wave_array = WW3.t02.values
        sig_wave_array = WW3.hs.values
        whitecap_array = WW3.wcc.values

        # load dates from netcdf file
        datelist = WW3.time.values.astype('datetime64[s]').astype(datetime)

        del(WW3)

        # create an interpolated array of the 1st slice so you can stack it onto the rest
        mean_wave = np.expand_dims(griddata(points, mean_wave_array[0].ravel(), xi, method='linear'),0)
        sig_wave = np.expand_dims(griddata(points, sig_wave_array[0].ravel(), xi, method='linear'),0)
        whitecap = np.expand_dims(griddata(points, whitecap_array[0].ravel(), xi, method='linear'),0)
        
        # interpolate and stack the rest of the grids
        for grid in range(1, mean_wave_array.shape[0]):
            interp_mean_wave = griddata(points, mean_wave_array[grid].ravel(), xi, method='linear')
            mean_wave = np.vstack((mean_wave, np.expand_dims(interp_mean_wave,0)))
            
            interp_sig_wave = griddata(points, sig_wave_array[grid].ravel(), xi, method='linear')
            sig_wave = np.vstack((sig_wave, np.expand_dims(interp_sig_wave,0)))
            
            interp_whitecap = griddata(points, whitecap_array[grid].ravel(), xi, method='linear')
            whitecap = np.vstack((whitecap, np.expand_dims(interp_whitecap,0)))
        
        # cut off grid edges
        mean_wave = mean_wave[...,:,1:897:,1:397]
        sig_wave = sig_wave[...,:,1:897:,1:397]
        whitecap = whitecap[...,:,1:897:,1:397]

        # transpose grid (rotate 90 clockwise)
        mean_wave = np.transpose(mean_wave, [0, 2, 1])
        sig_wave = np.transpose(sig_wave, [0, 2, 1])
        whitecap = np.transpose(whitecap, [0, 2, 1])

        # convert nans to 0s and set datatype to float64
        mean_wave = np.nan_to_num(mean_wave).astype('float64')
        sig_wave = np.nan_to_num(sig_wave).astype('float64')
        whitecap = np.nan_to_num(whitecap).astype('float64')

        # make list of time arrays
        datearrays = []
        for date in datelist:
            datearrays.append(
                np.array([date.year, date.month, date.day, date.hour, date.minute, date.second]).astype('float64')
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
                'Maximum' : np.array([float(datearray[0])]),
                'Minimum' : np.array([-0.]),
                'Units' : b'YYYY/MM/DD HH:MM:SS'
                } 
            dset.attrs.update(metadata)

        # write mean wave period to hdf5
        for i, dataset in enumerate(mean_wave):
            child_name = 'mean wave period_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = mwp.create_dataset(
                child_name,
                shape = (396, 896),
                data = dataset,
                chunks = (396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([100000.]),
                'Minimum' : np.array([0.]),
                'Units' : b's'
                }
            dset.attrs.update(metadata)

        # write significant wave height to hdf5
        for i, dataset in enumerate(sig_wave):
            child_name = 'significant wave height_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = swh.create_dataset(
                child_name,
                shape = (396, 896),
                data = dataset,
                chunks = (396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([100.]),
                'Minimum' : np.array([-100.]),
                'Units' : b'm'
                }
            dset.attrs.update(metadata)
    
        # write whitecap coverage to hdf5
        for i, dataset in enumerate(whitecap):
            child_name = 'whitecap coverage_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = wc.create_dataset(
                child_name,
                shape = (396, 896),
                data = dataset,
                chunks = (396, 896),
                compression = 'gzip',
                compression_opts = compression_level
                )
            metadata = {
                'FillValue' : np.array([0.]),
                'Maximum' : np.array([1.]),
                'Minimum' : np.array([0.]),
                'Units' : b'1'
                }
            dset.attrs.update(metadata)

        # update the accumulator
        attr_counter = attr_counter + mean_wave.shape[0]
        
        # clear memory
        del(mean_wave); del(sig_wave); del(whitecap)

    f.close()
    return

@timer
def create_e3t_hdf5(e3t_files, dirname, compression_level = 1, mask = mask):
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
    e3t_group = f.create_group('/Results/e3t')

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
        e3t = e3t*mask
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
            dset = e3t_group.create_dataset(
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

def init():
    # input start time
    def timestart():
        print('Date range entry is not inclusive of the end date i.e. [start date, end date)')
        timestart =  input('\nEnter the start time e.g. 2015 Jan 1:\n--> ')
        try:
            parse(timestart)
        except ValueError:
            print('Invalid input. Check format and enter correctly')
            timestart()
        return timestart
        
    # input end time. This must be day after required range as upper bound is not inlcuded
    def timeend():
        timeend = input('\nEnter the end time e.g. 2015 Jan 2:\n--> ')
        try:
            parse(timeend)
        except ValueError:
            print('\nInvalid input. Check format and enter correctly\n')
            timeend()
        return timeend
    
    # select which components to build
    def run_choice():
        runsdict = {1: 'All', 2: 'Salish SeaCast ONLY', 3: 'HRDPS ONLY', 4: 'WW3 ONLY'}
        try: 
            runs = int(input('\nRun: \n1) All \n2) Salish SeaCast ONLY \n3) HRDPS ONLY \n4) WW3 ONLY?\n--> '))
        except ValueError:
            print('\nSelect a valid run option\n')
            run_choice()
        if runs not in runsdict:
            print('\nSelect a valid run option\n')
            run_choice()
        ask = input(f'\nProceed with generating MOHID input files for {runsdict[runs]} from {timestart} to {timeend}? (yes/no)\n--> ')
        if ask in ['y', 'yes', 'YES', 'Y']:
            run(runs)
        else:
            print('\nAborted')

    # make the run selected
    @timer
    def run(runs):
        if runs == 1:
            salishseacast = salishseacast_paths(
                timestart, timeend, nemoinput, outpath, compression_level = 1
                )
            hrdps = hrdps_paths(
                timestart, timeend, hdinput, outpath, compression_level = 1
                )
            ww3 = ww3_paths(
                timestart, timeend, wwinput, outpath, compression_level = 1
                )
            if (salishseacast or hrdps or ww3) is False:
                print('\nAborted')
                return 
            create_currents_hdf5(*salishseacast[0])
            create_t_hdf5(*salishseacast[1])
            create_e3t_hdf5(*salishseacast[2])
            create_winds_hdf5(*hrdps)
            create_ww3_hdf5(*ww3)

        if runs == 2:
            salishseacast = salishseacast_paths(
                timestart, timeend, nemoinput, outpath, compression_level = 1
                )
            if salishseacast is False:
                print('\nAborted')
                return
            create_currents_hdf5(*salishseacast[0])
            create_t_hdf5(*salishseacast[1])
            create_e3t_hdf5(*salishseacast[2])

        if runs == 3:
            hrdps = hrdps_paths(
                timestart, timeend, hdinput, outpath, compression_level = 1
                )
            if hrdps is False:
                print('\nAborted')
                return
            create_winds_hdf5(*hrdps)
        
        if runs == 4:
            ww3 = ww3_paths(
                timestart, timeend, wwinput, outpath, compression_level = 1
                )
            if ww3 is False:
                print('\nAborted')
                return
            create_ww3_hdf5(*ww3)
            
        print('\nAll done')
    
    # user input date range
    timestart = timestart()
    timeend = timeend()
        
    # validate date range
    if np.diff([parse(t) for t in [timestart, timeend]])[0].days <= 0:
        print(f'\nInvalid input. Start time {timestart} comes after end time {timeend}.\n')
        init()
    else:
        run_choice()
    return

e3t_files = salishseacast_paths(
                '1 December 2017', '8 December 2017', nemoinput, outpath, compression_level = 1
                )[2]
create_e3t_hdf5(*e3t_files)
