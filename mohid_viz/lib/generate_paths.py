import os
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse

def salishseacast_paths(timestart, timeend, parameter, nemo_path='/results2/SalishSea/nowcast-green.201806/'):
    """Generate paths for Salish Seacast forcing 

    :arg timestart: date from when to start concatenating
    :type string: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type string: :py:class:'str'

    :arg parameter: string, one of 'U', 'V', 'W', 'T', 'e3t'

    :arg path: optional, path of input files
    :type string: :py:class:'str'

    :returns paths: list of file path strings
    :rtype: :py:class:`list'
    """
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]
    assert(parameter in ["U", "V", "W", "T", "e3t"]), "Invalid parameter. Parameter must be one of: ['U', 'V', 'W', 'T', 'e3t']"
    if parameter is "e3t":
        grid_or_carp = "carp"
        parameter = "T"
    else:
        grid_or_carp = "grid"
    # append all filename strings within daterange to lists
    paths = []
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days = day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d')
        # check if file exists. exit if it does not. add path to list if it does.
        path = f'{nemo_path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_{grid_or_carp}_{parameter}.nc'
        assert(os.path.exists(path)), f'File {path} not found. Check Directory and/or Date Range.'
        paths.append(path)
    return paths

def hrdps_paths(timestart, timeend, hrdps_path='/results/forcing/atmospheric/GEM2.5/operational/'):
    """Generate wind input file paths
    
    :arg timestart: date from when to start concatenating
    :type string: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type string: :py:class:'str'

    :arg hrdps_path: optional, path of input files
    :type string: :py:class:'str'

    :returns wind_files: list of file path strings
    :rtype: :py:class:`list'
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
        wind_path = f'{hrdps_path}ops_y{year}m{month}d{day}.nc'
        assert(os.path.exists(wind_path)), f'File {wind_path} not found. Check Directory and/or Date Range.'
        wind_files.append(wind_path)
    return wind_files

def ww3_paths(timestart, timeend, ww3_path='/opp/wwatch3/nowcast/', return_not_available=False):
    """Generate Wave Watch 3 input files paths

    :arg timestart: date from when to start concatenating
    :type string: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type string: :py:class:'str'

    :arg ww3_path: optional, path of input files
    :type string: :py:class:'str'

    arg return_not_activated: Optional bool. When True, it will return a list of missing files
    :type boolean: :py:class:'bool'

    :returns wave_files: list of file path strings
    :rtype: :py:class:`list'

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
        wave_path = f'{ww3_path}{day}{monthnm}{year}/SoG_ww3_fields_{datestr2}_{datestr2}.nc'
        if return_not_available is False:
            assert(os.path.exists(wave_path)), f'File {wave_path} not found. Check Directory and/or Date Range.'
            wave_files.append(wave_path)
        else:
            if not os.path.exists(wave_path):
                wave_files.append(wave_path)
    return wave_files

