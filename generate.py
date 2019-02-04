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

"""Locates and concatenates datafiles needed for MOHID based on user input time range
"""

import os
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np
import errno
import time

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
    :type array: :py:class:'int'

    :returns: time elapsed in HH:MM:SS format
    :rtype: :py:class:`str'
    """
    hours = int(time/3600)
    mins = int((time - (hours*3600))/60)
    secs = int((time - (3600 * hours) - (mins *60)))
    return '{}:{}:{}'.format(hours, mins, secs)


def generate_paths_NEMO(timestart, timeend, path, outpath):
    """Concatenate NEMO U, V, W and T files for MOHID.

    :arg timestart: date from when to start concatenating
    :type array: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type array: :py:class:'str'

    :arg path: path of input files
    :type array: :py:class:'str'

    :arg outpath: path for output files
    :type array: :py:class:'str'

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]
    U_files = []
    V_files = []
    W_files = []
    T_files = []
    # string: output folder name with date ranges used. end date will be lower by a day than timeend because datasets only go until midnight
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())

    # append all filename strings within daterange to lists
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d')
        U_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_U.nc')
        V_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_V.nc')
        W_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_W.nc')
        T_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_T.nc')
    shell_U = 'ncrcat '
    # concatenate all U parameter filename strings to create shell command
    for file in U_files:
        shell_U = shell_U + file + ' '
    # concatenate output filename and directory to end of shell command
    shell_U = shell_U + f'{outpath}nowcast-green/{folder}/' + 'U.nc'
    
    # concatenate all V parameter filename strings to create shell command
    shell_V = 'ncrcat '
    for file in V_files:
        shell_V = shell_V + file + ' '
    # concatenate output filename and directory to end of shell command
    shell_V = shell_V + f'{outpath}nowcast-green/{folder}/' + 'V.nc'

    # concatenate all W parameter filename strings to create shell command
    shell_W = 'ncrcat '
    for file in W_files:
        shell_W = shell_W + file + ' '
    # concatenate output filename and directory to end of shell command
    shell_W = shell_W + f'{outpath}nowcast-green/{folder}/' + 'W.nc'

    # concatenate all T parameter filename strings to create shell command
    shell_T = 'ncrcat '
    for file in T_files:
        shell_T = shell_T + file + ' '
    # concatenate output filename and directory to end of shell command
    shell_T = shell_T + f'{outpath}nowcast-green/{folder}/' + 'T.nc'

    # create output directory
    dirname = f'{outpath}nowcast-green/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    # run shell scripts to concatenate netcdf files
    for line in [shell_U, shell_V, shell_W, shell_T]:
        #print(line) # to test
        os.system(line)
    return None


def generate_paths_HRDPS(timestart, timeend, path, outpath):
    """Concatenate HRDPS files for MOHID.

    :arg timestart: date from when to start concatenating
    :type array: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type array: :py:class:'str'

    :arg path: path of input files
    :type array: :py:class:'str'

    :arg outpath: path for output files
    :type array: :py:class:'str'

    :returns: None
    :rtype: :py:class:`NoneType'
    """
    # generate list of dates from daterange given
    daterange = [parse(t) for t in [timestart, timeend]]
    wind_files = []
    # string: output folder name with date ranges used. end date will be lower by a day than timeend because datasets only go until midnight
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    # append all filename strings within daterange to list
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        month = datestamp.month
        if month < 10:
            month = f'0{str(month)}'
        day = datestamp.day
        if day < 10:
            day = f'0{str(day)}'
        year = str(datestamp.year)
        wind_files.append(f'{path}ops_y{year}m{month}d{day}.nc')
    shell_wind = 'ncrcat '
    # concatenate all GEM filename strings to create shell command
    for file in wind_files:
        shell_wind = shell_wind + file + ' '
    # concatenate output filename and directory to end of shell command
    shell_wind = shell_wind +  f'{outpath}hrdps/{folder}/'  + 'GEM.nc'

    # create output directory
    dirname = f'{outpath}hrdps/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    #print(shell_wind) # to test 
    # run shell script to concatenate netcdf files
    os.system(shell_wind)
    return None


def generate_paths_WW3(timestart, timeend, path, outpath):
    """Concatenate WW3 files for MOHID.

    :arg timestart: date from when to start concatenating
    :type array: :py:class:'str'

    :arg timeend: date at which to stop concatenating
    :type array: :py:class:'str'

    :arg path: path of input files
    :type array: :py:class:'str'

    :arg outpath: path for output files
    :type array: :py:class:'str'

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
        wave_files.append(f'{path}{day}{monthnm}{year}/SoG_ww3_fields_{datestr2}_{datestr2}.nc')
    shell_wave = 'ncrcat '
    # concatenate all WW3 filename strings to create shell command
    for file in wave_files:
        shell_wave = shell_wave + file + ' '
    # concatenate output filename and directory to end of shell command
    shell_wave = shell_wave + f'{outpath}ww3/{folder}/' + 'WW3.nc'
    # create output directory
    dirname = f'{outpath}ww3/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    
    #print(shell_wave)
    # run shell script to concatenate netcdf files
    os.system(shell_wave)
    return None

# input start time
timestart = input('\nEnter the start time in the format |year month day| e.g. 2015 Jan 1:\n--> ')

# input end time. This must be day after required range as upper bound is not inlcuded
timeend = input('\nEnter the end time in the format |year month day| e.g. 2015 Jan 1:\n--> ')

# what parts to run
runs = int(input('\nRun: \n1) All \n2) NEMO ONLY \n3) HRDPS ONLY \n4) WW3 ONLY ?\n--> '))
runsdict = {1: 'All', 2: 'NEMO ONLY', 3: 'HRDPS ONLY', 4: 'WW3 ONLY'}

# user confirmation
ask = input(f'\nProceed with concatenating {runsdict[runs]} from {timestart} to {timeend}?\n--> ')
beganat = time.time()

# run according to user inputs
if ask in ['y', 'yes', 'YES', 'Y']:
    if runs == 1:
        generate_paths_NEMO(timestart,timeend, nemoinput, outpath)
        nemotime = time.time() - beganat
        print('\n\nNEMO done\n')
        print('Time elapsed: {}\n\n'.format(conv_time(nemotime)))
        generate_paths_HRDPS(timestart,timeend, hdinput, outpath)
        hdtime = time.time() - beganat
        print('\nHRDPS done\n')
        print('Time elapsed: {}\n'.format(conv_time(hdtime-nemotime)))
        print('Total time elapsed: {}\n\n'.format(conv_time(hdtime)))
        generate_paths_WW3(timestart,timeend, wwinput, outpath)
        wwtime = time.time() - beganat
        print('\nWW3 done\n')
        print('Time elapsed: {}\n'.format(conv_time(wwtime-hdtime)))
        print('Total time elapsed: {}\n'.format(conv_time(wwtime)))
    if runs == 2:
        generate_paths_NEMO(timestart,timeend, nemoinput, outpath)

    if runs == 3:
        generate_paths_HRDPS(timestart,timeend, hdinput, outpath)

    if runs == 4:
        generate_paths_WW3(timestart,timeend, wwinput, outpath)
else:
    print('\nAborted')
print('All done\n')
print('Time elapsed: {}'.format(conv_time(time.time()-beganat)))
