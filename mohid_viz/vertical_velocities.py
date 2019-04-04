import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from datetime import datetime, timedelta
from dateutil.parser import parse

def _convert_timestamp(time):
    """Convert datetime.datetime to string in datetime64[s] format
    :arg time: datetime.datetime object
    :return datetime64: str in datetime64[s] format
    """
    year, month, day, hour, minute, second = str(time.year), str(time.month), str(time.day), str(time.hour), str(time.minute), str(time.second)
    if len(month) < 2:
        month = '0' + month
    if len(day) < 2:
        day = '0' + day
    if len(hour) < 2:
        hour = '0' + hour
    if len(minute) < 2:
        minute = '0' + minute
    if len(second) < 2:
        second = '0' + second
    datetime64 = '{}-{}-{}T{}:{}:{}'.format(year, month, day, hour, minute, second)
    return datetime64

class vertical_velocities():
    def __init__(self, xr_path, begin=None, end=None, top=None, bottom=None, plot_max=None):
        """initialise based on instance attributes and create plot
        arg xr_path: str, path of netcdf file
        arg begin: beginning time, optional 
        arg end:  end time, optional
        arg top: index of top depth layer to slice by, optional. int [0,39]
        arg bottom: index of bottom depth layer to slice by, optional. int [0,39]
        arg plot_max: still working on this one
        """
        xr_file = xr.open_dataset(xr_path)
        self.xr_file = xr_file
        self.begin_time = begin
        self.end_time = end
        self.__time_slice__()
        self.top_depth = top
        self.bottom_depth = bottom    
        self.__depth_slice__()
        self.__slice__()
        self.__plot__()
        
    def data(self):
        return self.current_view
    
    def __find_bottom__(self):
        i=0
        for value in np.flip(self.xr_file.vovecrtz.isel(time_counter = 0).values):
            if value != 0:
                self.bottom_depth = 39-i
            else:
                i=i+1

    def __plot__(self):
        self.current_view.T.plot(cmap = 'RdBu')
        plt.gca().invert_yaxis()
    
    def __slice__(self):
        attrs = (self.xr_file, self.begin_time, self.end_time, self.top_depth, self.bottom_depth)
        xr_file, begin_time, end_time, top_depth, bottom_depth = attrs
        self.current_view = xr_file.vovecrtz.sel(
            time_counter = slice(begin_time, end_time)
        ).isel(
            depthw = slice(top_depth, bottom_depth)
        )
    
    def __reset__(self):
        self.begin_time = None
        self.end_time = None
        self.top_depth = None
        self.bottom_depth = None
    
    def __time_slice__(self):
        begin, end = self.begin_time, self.end_time
        if (begin is None) and (end is None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = str(self.xr_file.time_counter[-1].values)
        elif (begin is None) and (end is not None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = _convert_timestamp(parse(end))
        elif (begin is not None) and (end is None):
            self.begin_time = _convert_timestamp(parse(begin))
            self.end_time = str(self.xr_file.time_counter[-1].values)
        else:
            daterange = [parse(t) for t in [begin, end]]
            assert(np.diff(daterange)[0].days >= 0), "Invalid Date Range"
            self.begin_time = _convert_timestamp(daterange[0])
            self.end_time = _convert_timestamp(daterange[1])
    
    def __depth_slice__(self):      
        begin, end = self.top_depth, self.bottom_depth
        assert((type(i) is int or None) for i in (begin, end)), "Depth slices are integers [0,39], or None"
        if (begin is None) and (end is None):
            self.top_depth = 0
            self.__find_bottom__()
        elif (begin is None):
            assert(0 <= end <= 39), "bottom_depth_depth must be in [0,39]"
            self.top_depth = 0
        elif (end is None):
            assert(0 <= begin <= 39), "top_depth must be in [0,39]"
            self.__find_bottom__()
        else:
            assert(end-begin >= 0), "End must be larger than begin"

    def __plot_max__(self):
        xrfile = self.xr_file
        data = np.abs(xrfile.vovecrtz.values)
        time = xrfile.time_counter.values
        depths = xrfile.depthw.values
        max_depths = []
        for i in data:
            max_depths.append(float(depths[np.where(i == i.max())][0]))
        plt.plot(time, max_depths)

class velocity():
    def __init__(self, xr_path, begin=None, end=None, top=None, bottom=None):         
        xr_file = xr.open_dataset(xr_path)
        self.xr_file = xr_file
        self.begin_time = begin
        self.end_time = end
        self.__time_slice__()
        self.top_depth = top
        self.bottom_depth = bottom
        self.__depth_slice__()
        self.__slice__()
        self.__plot__()
        
    def data(self):
        return self.current_view
        
    def __plot__(self):
        self.current_view.T.plot(cmap = 'RdBu')
        plt.gca().invert_yaxis()
    
    def __find_bottom__(self):
        i=0
        try:
            for value in np.flip(self.xr_file.vozocrtx.isel(time_counter = 0).values):
                if value != 0:
                    self.bottom_depth = 39-i
                else:
                    i=i+1
        except AttributeError:
            for value in np.flip(self.xr_file.vomecrty.isel(time_counter = 0).values):
                if value != 0:
                    self.bottom_depth = 39-i
                else:
                    i=i+1            
    
    def __slice__(self):
        attrs = (self.xr_file, self.begin_time, self.end_time, self.top_depth, self.bottom_depth)
        xr_file, begin_time, end_time, top_depth, bottom_depth = attrs
        try:
            self.current_view = xr_file.vozocrtx.sel(
                time_counter = slice(begin_time, end_time)
            ).isel(
                depthu = slice(top_depth, bottom_depth)
            )
        except AttributeError:
                        self.current_view = xr_file.vomecrty.sel(
                time_counter = slice(begin_time, end_time)
            ).isel(
                depthv = slice(top_depth, bottom_depth)
            )
    
    def __reset__(self):
        self.begin_time = None
        self.end_time = None
        self.top_depth = None
        self.bottom_depth = None
    
    def __time_slice__(self):
        begin, end = self.begin_time, self.end_time
        if (begin is None) and (end is None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = str(self.xr_file.time_counter[-1].values)
        elif (begin is None) and (end is not None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = _convert_timestamp(parse(end))
        elif (begin is not None) and (end is None):
            self.begin_time = _convert_timestamp(parse(begin))
            self.end_time = str(self.xr_file.time_counter[-1].values)
        else:
            daterange = [parse(t) for t in [begin, end]]
            assert(np.diff(daterange)[0].days >= 0), "Invalid Date Range"
            self.begin_time = _convert_timestamp(daterange[0])
            self.end_time = _convert_timestamp(daterange[1])
    
    def __depth_slice__(self):      
        begin, end = self.top_depth, self.bottom_depth
        assert((type(i) is int or None) for i in (begin, end)), "Depth slices are integers [0,39], or None"
        if (begin is None) and (end is None):
            self.top_depth = 0
            self.__find_bottom__()
        elif (begin is None):
            assert(0 <= end <= 39), "bottom_depth_depth must be in [0,39]"
            self.top_depth = 0
        elif (end is None):
            assert(0 <= begin <= 39), "top_depth must be in [0,39]"
            self.__find_bottom__()
        else:
            assert(end-begin >= 0), "End must be larger than begin"

class salinity():
    def __init__(self, xr_path, begin=None, end=None, top=None, bottom=None):         
        xr_file = xr.open_dataset(xr_path)
        self.xr_file = xr_file
        self.begin_time = begin
        self.end_time = end
        self.__time_slice__()
        self.top_depth = top
        self.bottom_depth = bottom
        self.__depth_slice__()
        self.__slice__()
        self.__plot__()
        
    def __plot__(self):
        self.data.T.plot(cmap = cmocean.cm.haline)
        plt.gca().invert_yaxis()
    
    def __find_bottom__(self):
        i=0
        for value in np.flip(self.xr_file.vosaline.isel(time_counter = 0).values):
            if value != 0:
                self.bottom_depth = 39-i
            else:
                i=i+1
    
    def __slice__(self):
        attrs = (self.xr_file, self.begin_time, self.end_time, self.top_depth, self.bottom_depth)
        xr_file, begin_time, end_time, top_depth, bottom_depth = attrs
        self.data = xr_file.vosaline.sel(
            time_counter = slice(begin_time, end_time)
        ).isel(
            deptht = slice(top_depth, bottom_depth)
        )
    
    def __reset__(self):
        self.begin_time = None
        self.end_time = None
        self.top_depth = None
        self.bottom_depth = None
    
    def __time_slice__(self):
        begin, end = self.begin_time, self.end_time
        if (begin is None) and (end is None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = str(self.xr_file.time_counter[-1].values)
        elif (begin is None) and (end is not None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = _convert_timestamp(parse(end))
        elif (begin is not None) and (end is None):
            self.begin_time = _convert_timestamp(parse(begin))
            self.end_time = str(self.xr_file.time_counter[-1].values)
        else:
            daterange = [parse(t) for t in [begin, end]]
            assert(np.diff(daterange)[0].days >= 0), "Invalid Date Range"
            self.begin_time = _convert_timestamp(daterange[0])
            self.end_time = _convert_timestamp(daterange[1])
    
    def __depth_slice__(self):      
        begin, end = self.top_depth, self.bottom_depth
        assert((type(i) is int or None) for i in (begin, end)), "Depth slices are integers [0,39], or None"
        if (begin is None) and (end is None):
            self.top_depth = 0
            self.__find_bottom__()
        elif (begin is None):
            assert(0 <= end <= 39), "bottom_depth_depth must be in [0,39]"
            self.top_depth = 0
        elif (end is None):
            assert(0 <= begin <= 39), "top_depth must be in [0,39]"
            self.__find_bottom__()
        else:
            assert(end-begin >= 0), "End must be larger than begin"
    
    def delta(self):
        """Plot the difference between the top and bottom depth salinity selected
        """
        xr_file = self.data
        self.data = (xr_file.isel(deptht=0)-xr_file.isel(deptht=-1))
        self.data.plot()
        plt.xlim(self.begin_time, self.end_time)