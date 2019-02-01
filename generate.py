# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:08:11 2019

@author: Ashub
"""
# Generate netdcf files for MOHID

import os
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np
import errno
import time

nemoinput = '/results2/SalishSea/nowcast-green.201806/'
hdinput = '/results/forcing/atmospheric/GEM2.5/operational/'
geminput = '/opp/wwatch3/nowcast/'
outpath = '/results2/MIDOSS/forcing/SalishSeaCast/'

beganat = time.time()

def conv_time(time):
    hours = int(time/3600)
    mins = int((time - (hours*3600))/60)
    secs = int((time - (3600 * hours) - (mins *60)))
    return '{}:{}:{}'.format(hours, mins, secs)

def generate_paths_NEMO(timestart, timeend, path, outpath):
    daterange = [parse(t) for t in [timestart, timeend]]
    U_files = []
    V_files = []
    W_files = []
    T_files = []
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d')
        U_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_U.nc')
        V_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_V.nc')
        W_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_W.nc')
        T_files.append(f'{path}{datestr1}/SalishSea_1h_{datestr2}_{datestr2}_grid_T.nc')
    shell_U = 'ncrcat '
   # folder = f'{timestart}-{timeend}'.replace(" ", "")
    for file in U_files:
        shell_U = shell_U + file + ' '
    shell_U = shell_U + f'{outpath}nowcast-green/{folder}/' + 'U.nc'
    shell_V = 'ncrcat '
    for file in V_files:
        shell_V = shell_V + file + ' '
    shell_V = shell_V + f'{outpath}nowcast-green/{folder}/' + 'V.nc'
    shell_W = 'ncrcat '
    for file in W_files:
        shell_W = shell_W + file + ' '
    shell_W = shell_W + f'{outpath}nowcast-green/{folder}/' + 'W.nc'
    shell_T = 'ncrcat '
    for file in T_files:
        shell_T = shell_T + file + ' '
    shell_T = shell_T + f'{outpath}nowcast-green/{folder}/' + 'T.nc'
    dirname = f'{outpath}nowcast-green/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    for line in [shell_U, shell_V, shell_W, shell_T]:
        print(line)
        #os.system(line)


def generate_paths_HRDPS(timestart, timeend, path, outpath):
    daterange = [parse(t) for t in [timestart, timeend]]
    wind_files = []
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        datestr1 = datestamp.strftime('%d%b%y').lower()
        #datestr2 = datestamp.strftime('%Y%m%d')
        month = datestamp.month
        if month < 10:
            month = f'0{str(month)}'
        day = datestamp.day
        if day < 10:
            day = f'0{str(day)}'
        year = str(datestamp.year)
        wind_files.append(f'{path}ops_y{year}m{month}d{day}.nc')
    shell_wind = 'ncrcat '
    for file in wind_files:
        shell_wind = shell_wind + file + ' '
    #folder = f'{timestart}-{timeend}'.replace(" ", "")
    shell_wind = shell_wind +  f'{outpath}hdrps/{folder}/'  + 'GEM.nc'
    dirname = f'{outpath}hdrps/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    print(shell_wind)
    #os.system(shell_wind)

def generate_paths_WW3(timestart, timeend, path, outpath):
    months = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5 : 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9 : 'sep', 10: 'oct', 11 :'nov',12: 'dec' }
    daterange = [parse(t) for t in [timestart, timeend]]
    wave_files = []
    folder = str(datetime(parse(timestart).year, parse(timestart).month, parse(timestart).day).strftime('%d%b%y').lower()) + '-' + str(datetime(parse(timeend).year, parse(timeend).month, parse(timeend).day-1).strftime('%d%b%y').lower())
    for day in range(np.diff(daterange)[0].days):
        datestamp = daterange[0] + timedelta(days=day)
        #datestr1 = datestamp.strftime('%d%b%y').lower()
        datestr2 = datestamp.strftime('%Y%m%d').lower()
        monthnm = months[datestamp.month]
        day = datestamp.day
        if day < 10:
            day = f'0{str(day)}'
        year = str(datestamp.year)[2:4]
        wave_files.append(f'{path}{day}{monthnm}{year}/SoG_ww3_fields_{datestr2}_{datestr2}.nc')
    shell_wave = 'ncrcat '
    for file in wave_files:
        shell_wave = shell_wave + file + ' '
    #folder = f'{timestart}-{timeend}'.replace(" ", "")
    shell_wave = shell_wave + f'{outpath}ww3/{folder}/' + 'WW3.nc'
    dirname = f'{outpath}ww3/{folder}/'
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    print(shell_wave)
    #os.system(shell_wave)


timestart = input('Enter the start time in the format |year month day| e.g. 2015 Jan 1:\n')
timeend = input('Enter the end time in the format |year month day| e.g. 2015 Jan 1:\n')
runs = int(input('Run: \n1) All \n2) NEMO ONLY \n3) HRDPS ONLY \n4) WW3 ONLY ?\n'))
runsdict = {1: 'All', 2: 'NEMO ONLY', 3: 'HRDPS ONLY', 4: 'WW3 ONLY'}
ask = input(f'Proceed with concatenating {runsdict[runs]} from {timestart} to {timeend}?\n')
if ask in ['y', 'yes', 'YES', 'Y']:
    if runs == 1:
        generate_paths_NEMO(timestart,timeend, nemoinput, outpath)
        nemotime = time.time() - beganat
        print('NEMO done\n')
        print('Time elapsed: {}\n'.format(conv_time(nemotime)))
        generate_paths_HRDPS(timestart,timeend, hdinput, outpath)
        hdtime = time.time() - beganat
        print('HRDPS done\n')
        print('Time elapsed: {}\n'.format(conv_time(hdtime-nemotime)))
        print('Total time elapsed: {}\n'.format(conv_time(hdtime)))
        generate_paths_WW3(timestart,timeend, hdinput, outpath)
        wwtime = time.time() - beganat
        print('HRDPS done')
        print('Time elapsed: {}'.format(conv_time(wwtime-hdtime)))
        print('Total time elapsed: {}\n'.format(conv_time(wwtime)))
    if runs == 2:
        generate_paths_NEMO(timestart,timeend, nemoinput, outpath)

    if runs == 3:
        generate_paths_HRDPS(timestart,timeend, hdinput, outpath)

    if runs == 4:
        generate_paths_WW3(timestart,timeend, hdinput, outpath)
else:
    print('\nAborted')
print('All done\n')
print('Time elapsed: {}'.format(conv_time(time.time()-beganat)))

