import errno
import forcing_paths
import mohid_interpolate
import h5py
import numpy
import os
import sys
import time
import xarray
from datetime import datetime, timedelta
from dateutil.parser import parse
import yaml


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

def mung_array(SSC_gridded_array, array_slice_type):
    """Transform an array containing SalishSeaCast-gridded data and transform it 
       into a MOHID-gridded array by:
         1) Cutting off the grid edges
         2) Transposing the X and Y axes
         3) Flipping the depth dimension, if it is present
         4) Converting the NaNs to 0
        
        :arg SSC_gridded_array: SalishSeaCast-gridded array
        :type numpy.ndarray: :py:class:'ndarray'
        
        :arg array_slice_type: str, one of '2D' or '3D'
        :type str: :py:class:'str'

        :return MOHID_gridded_array: MOHID-gridded array produced by applying operation 
                                     1-4 on SSC_gridded_array
        :type numpy.ndarray: :py:class:'ndarray'
    """
    shape = SSC_gridded_array.shape
    ndims = len(shape)
    assert(array_slice_type in  ('2D', '3D')), f"Invalid option {array_slice_type}. array_slice_type must be one of ('2D', '3D')"
    if array_slice_type is '2D':
        assert(ndims in (2,3)), f'The shape of the array given is {shape}, while the option chosen was {array_slice_type}'
        if ndims == 2:
            MOHID_gridded_array = SSC_gridded_array[1:897:,1:397]
            del(SSC_gridded_array)
            MOHID_gridded_array = numpy.transpose(MOHID_gridded_array, [1,0])
        else:
            MOHID_gridded_array = SSC_gridded_array[...,1:897:,1:397]
            del(SSC_gridded_array)
            MOHID_gridded_array = numpy.transpose(MOHID_gridded_array, [0,2,1])
    
    else:
        assert(ndims in (3,4)), f'The shape of the array given is {shape}, while the option chosen was {array_slice_type}'
        MOHID_gridded_array = SSC_gridded_array[...,1:897:,1:397]
        del(SSC_gridded_array)
        if ndims == 3:
            MOHID_gridded_array = numpy.transpose(MOHID_gridded_array, [0,2,1])
            MOHID_gridded_array = numpy.flip(MOHID_gridded_array, axis = 0)
        else:
            MOHID_gridded_array = numpy.transpose(MOHID_gridded_array, [0,1,3,2])
            MOHID_gridded_array = numpy.flip(MOHID_gridded_array, axis = 1)

    MOHID_gridded_array = numpy.nan_to_num(MOHID_gridded_array).astype('float64')

    return MOHID_gridded_array
        
def produce_datearray(datetimelist):
    """Produce a list of date arrays from a list of datetime objects
    """
    datearrays = [numpy.array(
        [d.year, d.month, d.day, d.hour, d.minute,d.second]
        ).astype('float64') for d in datetimelist]
    return datearrays

def unstagger_xarray(qty, index):
    """Interpolate u, v, or w component values to values at grid cell centres.
    
    Named indexing requires that input arrays are XArray DataArrays.

    :arg qty: u, v, or w component values
    :type qty: :py:class:`xarray.DataArray`
    
    :arg index: index name along which to centre
        (generally one of 'gridX', 'gridY', or 'depth')
    :type index: str

    :returns qty: u, v, or w component values at grid cell centres
    :rtype: :py:class:`xarray.DataArray`
    """

    qty = (qty + qty.shift(**{index: 1})) / 2

    return qty


def process_grid(file_paths, datatype, filename, groupname, compression_level, weighting_matrix_obj=None):
    accumulator = 1
    print(f'Writing {groupname} to {filename}...')
    for file_path in file_paths:
        data = xarray.open_dataset(file_path)
        if datatype in ('mean_wave_period','mean_wave_length','significant_wave_height', 'whitecap_coverage', 'stokesU', 'stokesV'):
            datetimelist = data.time.values.astype('datetime64[s]').astype(datetime)
        else:
            datetimelist = data.time_counter.values.astype('datetime64[s]').astype(datetime)
        datearrays = produce_datearray(datetimelist); del(datetimelist)
        if datatype is 'ocean_velocity_u':
            data = unstagger_xarray(data.vozocrtx, 'x').values
            data = mung_array(data, '3D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([5.]),
                'Minimum' : numpy.array([-5.]),
                'Units' : b'm/s'
                }
        elif datatype is 'ocean_velocity_v':
            data = unstagger_xarray(data.vomecrty, 'y').values
            data = mung_array(data, '3D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([5.]),
                'Minimum' : numpy.array([-5.]),
                'Units' : b'm/s'
                }
        elif datatype is 'ocean_velocity_w':
            data  = data.vovecrtz.values
            data = mung_array(data, '3D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([5.]),
                'Minimum' : numpy.array([-5.]),
                'Units' : b'm/s'
                }
        elif datatype is "vert_eddy_diff":
            data = data.vert_eddy_diff.values
            data = mung_array(data, "3D")
            metadata = {
                "FillValue": numpy.array([0.0]),
                "Maximum": numpy.array([5.0]),
                "Minimum": numpy.array([0.0]),
                "Units": b"m2/s",
            }
        elif datatype is 'salinity':
            data = data.vosaline.values
            data = mung_array(data, '3D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([100.]),
                'Minimum' : numpy.array([-100.]),
                'Units' : b'psu'
                }
        elif datatype is 'temperature':
            data = data.votemper.values
            data = mung_array(data, '3D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([100.]),
                'Minimum' : numpy.array([-100.]),
                'Units' : b'?C'
                }

        elif datatype is 'e3t':
            data = data.e3t.values
            data = mung_array(data, '3D')
            data = data*mask
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Units' : b'?C'
                }

        elif datatype is 'sea_surface_height':
            data = data.sossheig.values
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([5.]),
                'Minimum' : numpy.array([-5.]),
                'Units' : b'm'
                }
        elif datatype is 'wind_velocity_u':
            data = data.u_wind.values
            data = mohid_interpolate.hrdps(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([100.]),
                'Minimum' : numpy.array([-100.]),
                'Units' : b'm/s'
                }
        elif datatype is 'wind_velocity_v':
            data = data.v_wind.values
            data = mohid_interpolate.hrdps(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([100.]),
                'Minimum' : numpy.array([-100.]),
                'Units' : b'm/s'
                }
        elif datatype is 'mean_wave_period':
            data = data.t02.values
            data = mohid_interpolate.wavewatch(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([100000.]),
                'Minimum' : numpy.array([0.]),
                'Units' : b's'
                }
        elif datatype is "mean_wave_length":
            data = data.lm.values
            data = mohid_interpolate.wavewatch(data, weighting_matrix_obj)
            data = mung_array(data, "2D")
            metadata = {
                "FillValue": numpy.array([0.0]),
                "Maximum": numpy.array([3200.0]),
                "Minimum": numpy.array([0.0]),
                "Units": b"m",
            }
        elif datatype is 'significant_wave_height':
            data = data.hs.values
            data = mohid_interpolate.wavewatch(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([100.]),
                'Minimum' : numpy.array([-100.]),
                'Units' : b'm'
                }
        elif datatype is 'whitecap_coverage':
            data = data.wcc.values
            data = mohid_interpolate.wavewatch(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([1.]),
                'Minimum' : numpy.array([0.]),
                'Units' : b'1'
                }
        elif datatype is 'stokesU':
            data = data.uuss.values
            data = mohid_interpolate.wavewatch(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([9900.]),
                'Minimum' : numpy.array([-9900.]),
                'Units' : b'm s-1'
                }
        elif datatype is 'stokesV':
            data = data.vuss.values
            data = mohid_interpolate.wavewatch(data, weighting_matrix_obj)
            data = mung_array(data, '2D')
            metadata = {
                'FillValue' : numpy.array([0.]),
                'Maximum' : numpy.array([9900.]),
                'Minimum' : numpy.array([-9900.]),
                'Units' : b'm s-1'
                }
        write_grid(data, datearrays, metadata, filename, groupname, accumulator, compression_level)
        accumulator += len(datearrays)

# RuntimeError: Unable to create link (name already exists)

def write_grid(data, datearrays, metadata, filename, groupname, accumulator, compression_level):
    shape = data[0].shape
    with h5py.File(filename) as f:
        time_group = f.get('/Time')
        if time_group is None:
            time_group = f.create_group('/Time')
        data_group_path = f'/Results/{groupname}'
        data_group = f.get(data_group_path)
        if data_group is None:
            data_group = f.create_group(data_group_path)

        for i, datearray in enumerate(datearrays):
            numeric_attribute = ((5 - len(str(i + accumulator))) * '0') + str(i + accumulator)
            child_name = 'Time_' + numeric_attribute
            timestamp = time_group.get(child_name)
            if timestamp is None:
                dataset = time_group.create_dataset(
                    child_name,
                    shape = (6,),
                    data = datearray,
                    chunks = (6,),
                    compression = 'gzip',
                    compression_opts = compression_level
                    )
                time_metadata = {
                    'Maximum' : numpy.array(datearray[0]),
                    'Minimum' : numpy.array([-0.]),
                    'Units' : b'YYYY/MM/DD HH:MM:SS'
                    } 
                dataset.attrs.update(time_metadata)
            else:
                assert (numpy.asarray(timestamp) == datearray).all(), f'Time record {child_name} exists and does not match with {datearray}'
            
            child_name = groupname + '_' + numeric_attribute
            if data_group.get(child_name) is not None:
                print(f'Dataset already exists at {child_name}')
            else:
                dataset = data_group.create_dataset(
                    child_name,
                    shape = shape,
                    data = data[i],
                    chunks = shape,
                    compression = 'gzip',
                    compression_opts = compression_level
                    )
                dataset.attrs.update(metadata)

mask = mung_array(xarray.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSn3DMeshMaskV17-02').isel(time = 0).tmask.values, '3D')

@timer
def create_hdf5():
    script = sys.argv[0]
    yaml_filename = sys.argv[1]
    with open(yaml_filename, 'r') as f:
        run_description = yaml.safe_load(f)
    try:
        timerange = run_description['timerange']
        try:
            date_begin = timerange['date_begin']
            date_begin = parse(date_begin)
        except KeyError:
            print('date_begin not found'); return
        except ValueError:
            print(f'Invalid date_begin input: {date_begin}')

        try:
            date_end = timerange['date_end']
            date_end = parse(date_end)
        except KeyError:
            print('date_end not found'); return
        except ValueError:
            print(f'Invalid date_end input: {date_end}')

    except KeyError:
        print('timerange not found'); return

    if numpy.diff([date_begin, date_end])[0].days <= 0:
        print(f'\nInvalid input. date_begin comes after date_end')
        return

    try:
        paths = run_description['paths']
        salishseacast_path = paths.get('salishseacast')
        hrdps_path = paths.get('hrdps')
        wavewatch3_path = paths.get('wavewatch3')
        output_path = paths.get('output')
        wind_weights_path = paths.get('wind_weights')
        wave_weights_path = paths.get('wave_weights')

    except KeyError:
        print('No forcing file paths given')

    salish_seacast_forcing = run_description.get('salish_seacast_forcing')
    hrdps_forcing = run_description.get('hrdps_forcing')
    wavewatch3_forcing = run_description.get('wavewatch3_forcing')
    
    if salish_seacast_forcing is not None:
        currents_u = salish_seacast_forcing.get('currents').get('currents_u_hdf5_filename')
        currents_v = salish_seacast_forcing.get('currents').get('currents_v_hdf5_filename')
        vertical_velocity = salish_seacast_forcing.get('vertical_velocity').get('hdf5_filename')
        diffusivity = salish_seacast_forcing.get("diffusivity").get(
            "hdf5_filename"
        )
        salinity = salish_seacast_forcing.get('salinity').get('hdf5_filename')
        temperature = salish_seacast_forcing.get('temperature').get('hdf5_filename')
        sea_surface_height = salish_seacast_forcing.get('sea_surface_height').get('hdf5_filename')
        e3t = salish_seacast_forcing.get('e3t').get('hdf5_filename')

        for parameter in (currents_u, currents_v, vertical_velocity, diffusivity, salinity, temperature, sea_surface_height, e3t):
            if ((parameter is not None) and (salishseacast_path is None)):
                print('Path to SalishSeacast forcing not provided'); return
            elif (parameter is not None):
                if not os.path.exists(os.path.dirname(salishseacast_path)):
                    print(f'SalishSeaCast path {salishseacast_path} does not exist'); return
    
    wind_u = hrdps_forcing.get('winds').get('wind_u_hdf5_filename')
    wind_v = hrdps_forcing.get('winds').get('wind_v_hdf5_filename')
    
    for parameter in (wind_u, wind_v):
        if (parameter is not None):
            if (hrdps_path is None):
                print('Path to HRDPS forcing not provided'); return

            if not os.path.exists(os.path.dirname(hrdps_path)):
                print(f'HRDPS path {hrdps_path} does not exist'); return

            if (wind_weights_path is None):
                print('Path to wind interpolation weights file is not provided'); return
            
            if not os.path.exists(wind_weights_path):
                print(f'Path to wind interpolation weights file {wind_weights_path} does not exist'); return
            else:
                wind_weights = mohid_interpolate.weighting_matrix(wind_weights_path)

    whitecap_coverage = wavewatch3_forcing.get('whitecap_coverage').get('hdf5_filename')
    mean_wave_period = wavewatch3_forcing.get('mean_wave_period').get('hdf5_filename')
    mean_wave_length = wavewatch3_forcing.get("mean_wave_length").get("hdf5_filename")
    significant_wave_height = wavewatch3_forcing.get('significant_wave_height').get('hdf5_filename')
    stokesU = wavewatch3_forcing.get('stokesU').get('hdf5_filename')
    stokesV = wavewatch3_forcing.get('stokesV').get('hdf5_filename')

    for parameter in  (whitecap_coverage, mean_wave_period, mean_wave_length, significant_wave_height, stokesU, stokesV):
        if (parameter is not None):
            if (wavewatch3_path is None):
                print('Path to WaveWatch3 forcing not provided'); return

            if not os.path.exists(os.path.dirname(wavewatch3_path)):
                print(f'WaveWatch3 path {wavewatch3_forcing} does not exist'); return

            if (wave_weights_path is None):
                print('Path to wave interpolation weights file is not provided'); return
            
            if not os.path.exists(wave_weights_path):
                print(f'Path to wave interpolation weights file {wave_weights_path} does not exist'); return
            else:
                wave_weights = mohid_interpolate.weighting_matrix(wave_weights_path)

    try:
        compression_level = int(run_description.get('hdf5_compression_level', 4))
        if not (1 <= compression_level <= 9):
            print('Invalid compression level: {} provided. Compression level is int[1,9]. Default is 4'); return
    except ValueError:
        print('Invalid compression level: {} provided. Compression level is int[1,9]. Default is 4'); return

    # Make sure all the source files are available
    if salish_seacast_forcing is not None:
        if currents_u is not None:
            currents_u_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'grid_U')
            if not currents_u_list:
                return
        if currents_v is not None:
            currents_v_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'grid_V')
            if not currents_v_list:
                return
        if vertical_velocity is not None:
            vertical_velocity_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'grid_W')
            if not vertical_velocity_list:
                return
        if diffusivity is not None:
            diffusivity_list = forcing_paths.salishseacast_paths(
                date_begin, date_end, salishseacast_path, "grid_W"
            )
            if not diffusivity_list:
                return
        if temperature is not None:
            temperature_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'grid_T')
            if not temperature_list:
                return
        if salinity is not None:
            salinity_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'grid_T')
            if not salinity_list:
                return
        if sea_surface_height is not None:
            sea_surface_height_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'grid_T')
            if not sea_surface_height_list:
                return
        if e3t is not None:
            e3t_list = forcing_paths.salishseacast_paths(date_begin, date_end, salishseacast_path, 'carp_T')
            if not e3t_list:
                return
    if hrdps_forcing is not None:
        if wind_u is not None:
            wind_u_list = forcing_paths.hrdps_paths(date_begin, date_end, hrdps_path)
            if not wind_u_list:
                return
        if wind_v is not None:
            wind_v_list = forcing_paths.hrdps_paths(date_begin, date_end, hrdps_path)
            if not wind_v_list:
                return
    if wavewatch3_forcing is not None:
        if whitecap_coverage is not None:
            whitecap_coverage_list = forcing_paths.ww3_paths(date_begin, date_end, wavewatch3_path)
            if not whitecap_coverage_list:
                return
        if mean_wave_period is not None:
            mean_wave_period_list = forcing_paths.ww3_paths(date_begin, date_end, wavewatch3_path)
            if not mean_wave_period_list:
                return
        if mean_wave_length is not None:
            mean_wave_length_list = forcing_paths.ww3_paths(
                date_begin, date_end, wavewatch3_path
            )
            if not mean_wave_length_list:
                return
        if significant_wave_height is not None:
            significant_wave_height_list = forcing_paths.ww3_paths(date_begin, date_end, wavewatch3_path)
            if not significant_wave_height_list:
                return
        if stokesU is not None:
            stokesU_list = forcing_paths.ww3_paths(date_begin, date_end, wavewatch3_path)
            if not stokesU_list:
                return
        if stokesV is not None:
            stokesV_list = forcing_paths.ww3_paths(date_begin, date_end, wavewatch3_path)
            if not stokesV_list:
                return
    
    if output_path is None:
        print('No output file path provided'); return

    startfolder, endfolder = date_begin, date_end
    folder = str(
        datetime(startfolder.year, startfolder.month, startfolder.day).strftime('%d%b%y').lower()
        ) + '-' + str(
            datetime(endfolder.year, endfolder.month, endfolder.day).strftime('%d%b%y').lower()
            )

    # create output directory
    dirname = f'{output_path}MF0/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print(exc.errno)
                return

    print(f'\nOutput directory {dirname} created\n')


    # Now that everything is in place, we can start generating the .hdf5 files
    if salish_seacast_forcing is not None:
        if currents_u is not None:
            process_grid(currents_u_list, 'ocean_velocity_u', dirname+currents_u, 'velocity U', compression_level)

        if currents_v is not None:
            process_grid(currents_v_list, 'ocean_velocity_v', dirname+currents_v, 'velocity V', compression_level)

        if vertical_velocity is not None:
            process_grid(vertical_velocity_list, 'ocean_velocity_w', dirname+vertical_velocity, 'velocity W', compression_level)

        
        if diffusivity is not None:
            process_grid(
                diffusivity_list,
                "vert_eddy_diff",
                dirname + diffusivity,
                "Diffusivity",
                compression_level,
            )

        if temperature is not None:
            process_grid(temperature_list, 'temperature', dirname+temperature, 'temperature', compression_level)


        if salinity is not None:
            process_grid(salinity_list, 'salinity', dirname+salinity, 'salinity', compression_level)


        if sea_surface_height is not None:
            process_grid(sea_surface_height_list, 'sea_surface_height', dirname+sea_surface_height, 'water level', compression_level)


        if e3t is not None:
            process_grid(e3t_list, 'e3t', dirname+e3t, 'vvl', compression_level)
    if hrdps_forcing is not None:
        if wind_u is not None:
            process_grid(wind_u_list, 'wind_velocity_u', dirname+wind_u, 'wind velocity X', compression_level, wind_weights)


        if wind_v is not None:
            process_grid(wind_v_list, 'wind_velocity_v', dirname+wind_v, 'wind velocity Y', compression_level, wind_weights)


    if wavewatch3_forcing is not None:
        if whitecap_coverage is not None:
            process_grid(whitecap_coverage_list, 'whitecap_coverage', dirname+whitecap_coverage, 'whitecap coverage', compression_level, wave_weights)


        if mean_wave_period is not None:
            process_grid(mean_wave_period_list, 'mean_wave_period', dirname+mean_wave_period, 'mean wave period', compression_level, wave_weights)
       
        if mean_wave_length is not None:
            process_grid(
                mean_wave_length_list,
                "mean_wave_length",
                dirname + mean_wave_length,
                "mean wave length",
                compression_level,
                wave_weights,
            )

        if significant_wave_height is not None:
            process_grid(significant_wave_height_list, 'significant_wave_height', dirname+significant_wave_height, 'significant wave height', compression_level, wave_weights)


        if stokesU is not None:
            process_grid(stokesU_list, 'stokesU', dirname+stokesU, 'Stokes U', compression_level, wave_weights)

        if stokesV is not None:
            process_grid(stokesV_list, 'stokesV', dirname+stokesV, 'Stokes V', compression_level, wave_weights)
            

if __name__ == "__main__":
    create_hdf5()
