import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import lib.functions as funcs
from datetime import datetime, timedelta
from dateutil.parser import parse

class salinity():
    def __init__(
        self, xr_path,
        begin=None, end=None,
        top=None, bottom=None,
        title=None,
        max_delta=False,
        cmap=cmocean.cm.haline,
        verbose=False
        ):         
        xr_file = xr.open_dataset(xr_path)
        self.attrs = {
            "Start_time":"",
            "End_time":"",
            "Top_depth_layer":"",
            "Bottom_depth_layer":"",
            "Sea_floor_layer": ""}
        self.xr_file = xr_file
        # find out what attribute we are dealing with
        try:
            xr_file.vosaline
        except AttributeError:
            raise AttributeError('No salinity attribute exists in netcdf file')
        self.begin_time = begin
        self.end_time = end
        self.__time_slice__()
        self.top_depth = top
        self.bottom_depth = bottom
        self.__depth_slice__()
        self.__slice__()
        self.attrs["Start_time"]=self.begin_time
        self.attrs["End_time"]=self.end_time
        self.attrs["Top_depth_layer"]=self.top_depth
        self.attrs["Bottom_depth_layer"]=self.bottom_depth
        self.attrs["Sea_floor_layer"]=self.sea_floor
        self.__plot__(title,  max_delta, cmap)
        if verbose is True:
            print(self.attrs)
        
    def __plot__(self, title,  max_delta, cmap):
        self.data.T.plot(cmap = cmap)
        lat, lon = round(float(self.xr_file.nav_lat),5), round(float(self.xr_file.nav_lon),5)
        if max_delta is True:
            self.__max_delta__()
        if title is None:
            plt.title('Latitude: {}, Longitude: {}'.format(str(lat), str(lon)))
        else:
            plt.title = title
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
    
    def __time_slice__(self):
        begin, end = self.begin_time, self.end_time
        if (begin is None) and (end is None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = str(self.xr_file.time_counter[-1].values)
        elif (begin is None) and (end is not None):
            self.begin_time = str(self.xr_file.time_counter[0].values)
            self.end_time = funcs._convert_timestamp(parse(end))
        elif (begin is not None) and (end is None):
            self.begin_time = funcs._convert_timestamp(parse(begin))
            self.end_time = str(self.xr_file.time_counter[-1].values)
        else:
            daterange = [parse(t) for t in [begin, end]]
            assert(np.diff(daterange)[0].days >= 0), "Invalid Date Range"
            self.begin_time = funcs._convert_timestamp(daterange[0])
            self.end_time = funcs._convert_timestamp(daterange[1])
    
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
    
    def __max_delta__(self):
        """return raw plot data for depth of maximum delta
        """ 
        xr_dataarray = self.data
        values = xr_dataarray.values
        # time is axis 0, depth is axis 1
        difference = np.abs(np.diff(values, axis=1))
        times = xr_dataarray.time_counter.values
        data = (xr_dataarray.deptht.values[np.argmax(difference, axis=1)])
        plt.plot(times, data, 'k', linewidth=1)

    def diff(self):
        """Plot the difference between the top and bottom depth salinity selected
        """
        xr_file = self.data
        self.data = (xr_file.isel(deptht=0)-xr_file.isel(deptht=-1))
        self.data.plot()
        plt.xlim(self.begin_time, self.end_time)



