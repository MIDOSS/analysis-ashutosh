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
from scipy.interpolate import griddata

# NEMO input files directory
nemoinput = '/results2/SalishSea/nowcast-green.201806/'

# HRDPS input files directory
hdinput = '/results/forcing/atmospheric/GEM2.5/operational/'

# WW3 input files directory
wwinput = '/opp/wwatch3/nowcast/'

# Output filepath
outpath = '/results2/MIDOSS/forcing/SalishSeaCast/'


def timer(func):
    def f(*args, **kwargs):
        beganat = time.time()
        rv = func(*args, *kwargs)
        elapsed = time.time() - beganat
        hours = int(elapsed/3600)
        mins = int((elapsed - (hours*3600))/60)
        secs = int((elapsed - (3600 * hours) - (mins *60)))
        print('Time elapsed: {}:{}:{}'.format(hours, mins, secs))
        return rv
    return f

@timer
def generate_currents_hdf5(timestart, timeend, path, outpath, compression_level = 1):
    """Generate current forcing HDF5 input file for MOHID.

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

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]
    U_files = []
    V_files = []
    W_files = []
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
            # W files
        W_path = f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_W.nc'
        if not os.path.exists(W_path):
            print(f'File {W_path} not found. Check Directory and/or Date Range.')
            return
        W_files.append(W_path)
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
    f = h5py.File(f'{dirname}currents.hdf5', 'w')
    times = f.create_group('Time')
    velocity_u = f.create_group('/Results/velocity U')
    velocity_v = f.create_group('/Results/velocity V')
    velocity_w = f.create_group('/Results/velocity W')
    water_level = f.create_group('/Results/water level')
    
    attr_counter = 0
    number_of_files = len(U_files)
    bar = utilities.statusbar('Creating currents forcing file ...')
    for file_index in bar(range(number_of_files)):
        U_raw = xr.open_dataset(U_files[file_index])
        V_raw = xr.open_dataset(V_files[file_index])
        W_raw = xr.open_dataset(W_files[file_index])
        T_raw = xr.open_dataset(T_files[file_index])
        # assume all files have same time_counter markers
        datelist = U_raw.time_counter.values.astype('datetime64[s]').astype(datetime)
        # unstagger to move U, V to center of grid square
        U  = viz_tools.unstagger_xarray(U_raw.vozocrtx, 'x')
        V  = viz_tools.unstagger_xarray(V_raw.vomecrty, 'y')
        W  = W_raw.vovecrtz #!!! 
        # convert xarrays to numpy arrays and cut off grid edges
        U = U.values[...,:,1:897:,1:397]
        V = V.values[...,:,1:897:,1:397]
        current_w = W.values[...,:,1:897:,1:397]
        sea_surface = T_raw.sossheig.values[...,:,1:897:,1:397]
        # rotate currents to True North
        current_u, current_v = viz_tools.rotate_vel(U, V)
        # clear memory
        U, V = None, None
        # transpose grid (rotate 90 clockwise)
        current_u = np.transpose(current_u, [0,1,3,2])
        current_v = np.transpose(current_v, [0,1,3,2])
        current_w = np.transpose(current_w, [0,1,3,2])
        sea_surface = np.transpose(sea_surface, [0,2,1])
        # flip currents by depth dimension
        current_u = np.flip(current_u, axis = 1)
        current_v = np.flip(current_v, axis = 1)
        current_w = np.flip(current_w, axis = 1)
        # convert nans to 0's and set datatype to float64
        current_u = np.nan_to_num(current_u).astype('float64')
        current_v = np.nan_to_num(current_v).astype('float64')
        current_w = np.nan_to_num(current_w).astype('float64')
        sea_surface = np.nan_to_num(sea_surface).astype('float64')
        # make list of time arrays
        datearrays = []
        for date in datelist:
            datearrays.append(np.array([date.year, date.month, date.day, date.hour, date.minute, date.second]).astype('float64'))
        # write u wind values to hdf5
        for i in range(current_u.shape[0]):
            child_name = 'velocity U_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = velocity_u.create_dataset(child_name, shape = (40, 396, 896), data = current_u[i],chunks=(40, 396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([5.]), 'Minimum' : np.array([-5.]), 'Units' : b'm/s'}
            dset.attrs.update(metadata)
    
        # write v wind values to hdf5
        for i in range(current_v.shape[0]):
            child_name = 'velocity V_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = velocity_v.create_dataset(child_name, shape = (40, 396, 896), data = current_v[i],chunks=(40, 396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([5.]), 'Minimum' : np.array([-5.]), 'Units' : b'm/s'}
            dset.attrs.update(metadata)

        # write w wind values to hdf5
        for i in range(current_v.shape[0]):
            child_name = 'velocity W_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = velocity_w.create_dataset(child_name, shape = (40, 396, 896), data = current_w[i],chunks=(40, 396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([5.]), 'Minimum' : np.array([-5.]), 'Units' : b'm/s'}
            dset.attrs.update(metadata)
    
        # write  water level values to hdf5

        for i in range(sea_surface.shape[0]):
            child_name = 'water level_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = water_level.create_dataset(child_name, shape = (396, 896), data = sea_surface[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([5.]), 'Minimum' : np.array([-5.]), 'Units' : b'm'}
            dset.attrs.update(metadata)
    
        # write time values to hdf5

        for i in range(len(datearrays)):
            time_attr = 'Time_' + ((5 - len(str(i + attr_counter + 1))) * '0') + str(i + attr_counter + 1)
            dset = times.create_dataset(time_attr, shape = (6,), data = datearrays[i],chunks=(6,), compression = 'gzip', compression_opts = compression_level)
            metadata = {'Maximum' : np.array(datearrays[i][0]), 'Minimum' : np.array([-0.]), 'Units' : b'YYYY/MM/DD HH:MM:SS'} 
            dset.attrs.update(metadata)
        attr_counter = attr_counter + current_u.shape[0]
    f.close()
    return

@timer
def generate_winds(timestart, timeend, path, outpath, compression_level = 1):
    """Generate wind forcing HDF5 input file for MOHID.

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

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]
    wind_files = []

    # append all filename strings within daterange to list
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        month = datestamp.month
        if month < 10:
            month = f'0{str(month)}'
        day = datestamp.day
        if day < 10:
            day = f'0{str(day)}'
        year = str(datestamp.year)
        wind_path = f'{path}ops_y{year}m{month}d{day}.nc'
        if not os.path.exists(wind_path):
            print(f'File {wind_path} not found. Check Directory and/or Date Range.')
            return 
        wind_files.append(wind_path)
    print('\nAll source files found')
    # string: output folder name with date ranges used. end date will be lower by a day than timeend because datasets only go until midnight
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    dirname = f'{outpath}hrdps/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    print(f'\nOutput directory {dirname} created\n')
    # create hdf5 file and create tree structure
    f = h5py.File(f'{dirname}winds.hdf5', 'w')
    times = f.create_group('Time')
    windu = f.create_group('/Results/wind velocity X')
    windx = f.create_group('/Results/wind velocity Y')
    attr_counter = 0
    number_of_files = len(wind_files)
    bar = utilities.statusbar('Creating winds forcing file ...')
    for file_index in bar(range(number_of_files)):
        GEM = xr.open_dataset(wind_files[file_index])
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
        # create an interpolated array of the 1st slice so you can stack it onto the rest
        u_wind = np.expand_dims(griddata(points, GEM_u[0].ravel(), xi, method='cubic'),0)
        v_wind = np.expand_dims(griddata(points, GEM_v[0].ravel(), xi, method='cubic'),0)
        # create an interpolated array for the rest of the values
        for grid in range(1, GEM_u.shape[0]):
            interp_u = griddata(points, GEM_u[grid].ravel(), xi, method='cubic')
            u_wind = np.vstack((u_wind, np.expand_dims(interp_u,0)))
            interp_v = griddata(points, GEM_v[grid].ravel(), xi, method='cubic')
            v_wind = np.vstack((v_wind, np.expand_dims(interp_v,0)))
        u_wind = u_wind[...,:,1:897:,1:397]
        v_wind = v_wind[...,:,1:897:,1:397]
        u_wind = np.transpose(u_wind, [0,2,1])
        v_wind = np.transpose(v_wind, [0,2,1])
        u_wind = u_wind.astype('float64')
        v_wind = v_wind.astype('float64')
        datelist = GEM.time_counter.values.astype('datetime64[s]').astype(datetime.datetime)
        datearrays = []
        for date in datelist:
            datearrays.append(np.array([date.year, date.month, date.day, date.hour, date.minute, date.second]).astype('float64'))
        for i in range(len(datearrays)):
            child_name = 'Time_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset = times.create_dataset(child_name, shape = (6,), data = datearrays[i],chunks=(6,), compression = 'gzip', compression_opts = compression_level)
            metadata = {'Maximum' : np.array([float(datearrays[i][0])]), 'Minimum' : np.array([-0.]), 'Units' : b'YYYY/MM/DD HH:MM:SS'} 
            dset.attrs.update(metadata)
        for i in range(u_wind.shape[0]):
            child_name = 'wind velocity X_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset = windu.create_dataset(child_name, shape = (396, 896), data = u_wind[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([100.]), 'Minimum' : np.array([-100.]), 'Units' : b'm/s'}
            dset.attrs.update(metadata)
        for i in range(v_wind.shape[0]):
            child_name = 'wind velocity Y_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset =  windx.create_dataset(child_name, shape = (396, 896), data = v_wind[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([100.]), 'Minimum' : np.array([-100.]), 'Units' : b'm/s'}
            dset.attrs.update(metadata)
        attr_counter = attr_counter + u_wind.shape[0]
        f.close()
        return

@timer
def generate_ww3(timestart, timeend, path, outpath, compression_level = 1):
    """Generate wave surface forcing HDF5 input file for MOHID.

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

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # generate list of dates from daterange given
    months = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5 : 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9 : 'sep', 10: 'oct', 11 :'nov',12: 'dec' }
    daterange = [parse(t) for t in [timestart, timeend]]
    wave_files = []
    # string: output folder name with date ranges used. end date will be lower by a day than timeend because datasets only go until midnight
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    # append all filename strings within daterange to list
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
            return 
        wave_files.append(wave_path)
    print('\nAll source files found')
    # string: output folder name with date ranges used. end date will be lower by a day than timeend because datasets only go until midnight
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    dirname = f'{outpath}ww3/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    print(f'\nOutput directory {dirname} created\n')
    # create hdf5 file and create tree structure
    f = h5py.File(f'{dirname}ww3.hdf5', 'w')
    times = f.create_group('Time')
    mwp = f.create_group('/Results/mean wave period')
    swh = f.create_group('/Results/significant wave height')
    wc = f.create_group('/Results/whitecap coverage')
    attr_counter = 0
    number_of_files = len(wave_files)
    bar = utilities.statusbar('Creating WW3 forcing file ...')
    for file_index in bar(range(number_of_files)):
        WW3 = xr.open_dataset(wave_files[file_index])
        NEMO_grid = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSnBathymetryV17-02')
        # GEM data coordinates
        points = np.array([WW3.latitude.values.ravel(), WW3.longitude.values.ravel()-360]).T
        # NEMO lat lon grids tuple
        xi = (NEMO_grid.latitude.values, NEMO_grid.longitude.values)
        mean_wave_array = WW3.hs.values
        sig_wave_array = WW3.t02.values
        whitecap_array = WW3.wcc.values
        mean_wave = np.expand_dims(griddata(points, mean_wave_array[0].ravel(), xi, method='cubic'),0)
        sig_wave = np.expand_dims(griddata(points, sig_wave_array[0].ravel(), xi, method='cubic'),0)
        whitecap = np.expand_dims(griddata(points, whitecap_array[0].ravel(), xi, method='cubic'),0)
        for grid in range(1, mean_wave_array.shape[0]):
            interp_mean_wave = griddata(points, mean_wave_array[grid].ravel(), xi, method='cubic')
            mean_wave = np.vstack((mean_wave, np.expand_dims(interp_mean_wave,0)))
            interp_sig_wave = griddata(points, sig_wave_array[grid].ravel(), xi, method='cubic')
            sig_wave = np.vstack((sig_wave, np.expand_dims(interp_sig_wave,0)))
            interp_whitecap= griddata(points, whitecap_array[grid].ravel(), xi, method='cubic')
            whitecap = np.vstack((whitecap, np.expand_dims(interp_whitecap,0)))
        mean_wave = mean_wave[...,:,1:897:,1:397]
        sig_wave = sig_wave[...,:,1:897:,1:397]
        whitecap = whitecap[...,:,1:897:,1:397]
        mean_wave = np.transpose(mean_wave, [0, 2, 1])
        sig_wave = np.transpose(sig_wave, [0, 2, 1])
        whitecap = np.transpose(whitecap, [0, 2, 1])
        mean_wave = np.nan_to_num(mean_wave).astype('float64')
        sig_wave = np.nan_to_num(sig_wave).astype('float64')
        whitecap = np.nan_to_num(whitecap).astype('float64')
        datelist = WW3.time.values.astype('datetime64[s]').astype(datetime.datetime)
        datearrays = []
        for date in datelist:
            datearrays.append(np.array([date.year, date.month, date.day, date.hour, date.minute, date.second]).astype('float64'))
        for i in range(len(datearrays)):
            child_name = 'Time_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset = times.create_dataset(child_name, shape = (6,), data = datearrays[i],chunks=(6,), compression = 'gzip', compression_opts = compression_level)
            metadata = {'Maximum' : np.array([float(datearrays[i][0])]), 'Minimum' : np.array([-0.]), 'Units' : b'YYYY/MM/DD HH:MM:SS'} 
            dset.attrs.update(metadata)

        for i in range(mean_wave.shape[0]):
            child_name = 'mean wave period_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset = mwp.create_dataset(child_name, shape = (396, 896), data = mean_wave[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([100000.]), 'Minimum' : np.array([0.]), 'Units' : b's'}
            dset.attrs.update(metadata)

        for i in range(sig_wave.shape[0]):
            child_name = 'significant wave height_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset = swh.create_dataset(child_name, shape = (396, 896), data = sig_wave[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([100.]), 'Minimum' : np.array([-100.]), 'Units' : b'm'}
            dset.attrs.update(metadata)
    
        for i in range(whitecap.shape[0]):
            child_name = 'whitecap coverage_' + ((5 - len(str(i + 1))) * '0') + str(i + 1)
            dset = wc.create_dataset(child_name, shape = (396, 896), data = whitecap[i],chunks=(396, 896), compression = 'gzip', compression_opts = compression_level)
            metadata = {'FillValue' : np.array([0.]), 'Maximum' : np.array([1.]), 'Minimum' : np.array([0.]), 'Units' : b'1'}
            dset.attrs.update(metadata)

        attr_counter = attr_counter + mean_wave.shape[0]
    f.close()
    return
