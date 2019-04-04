import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from lib.functions import *

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
    
    def __find_bottom__(self, both=True):
        i=0
        for value in np.flip(self.xr_file.vosaline.isel(time_counter = 0).values):
            if value != 0:
                if both is False:
                    self.sea_floor = 39-i
                if both is True:
                    self.bottom_depth = 39-i 
                    self.sea_floor = 39-i
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
            self.__find_bottom__(both=True)
        elif (begin is None):
            assert(0 <= end <= 39), "bottom_depth_depth must be in [0,39]"
            self.top_depth = 0
            self.__find_bottom__(both=False)
        elif (end is None):
            assert(0 <= begin <= 39), "top_depth must be in [0,39]"
            self.__find_bottom__(both=True)
        else:
            assert(end-begin >= 0), "End must be larger than begin"
            self.__find_bottom__(both=False)
    def delta(self):
        """Plot the difference between the top and bottom depth salinity selected
        """
        xr_file = self.data
        self.data = (xr_file.isel(deptht=0)-xr_file.isel(deptht=-1))
        self.data.plot()
        plt.xlim(self.begin_time, self.end_time)
