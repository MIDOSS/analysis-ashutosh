import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import lib.functions as funcs
from datetime import datetime, timedelta
from dateutil.parser import parse

class velocity():
    def __init__(self, xr_path, begin=None, end=None, top=None, bottom=None, plot_max=None, cmap='RdBu', title=None, verbose=False):
        """initialise based on instance attributes and create plot
        arg xr_path: str, path of netcdf file
        arg begin: beginning time, optional 
        arg end:  end time, optional
        arg top: index of top depth layer to slice by, optional. int [0,39]
        arg bottom: index of bottom depth layer to slice by, optional. int [0,39]
        arg plot_max: still working on this one
        """    
        xr_file = xr.open_dataset(xr_path)
        self.attrs = {
            "Velocity": "",
            "Start_time":"",
            "End_time":"",
            "Top_depth_layer":"",
            "Bottom_depth_layer":"",
            "Sea_floor_layer": ""}
        self.xr_file = xr_file
        # find out what attribute we are dealing with
        try:
            xr_file.vovecrtz
            self.attribute = xr_file.vovecrtz
            self.velocity = 'W'
        except AttributeError:
            try:
                xr_file.vozocrtx
                self.attribute = xr_file.vozocrtx
                self.velocity = 'U'
            except AttributeError:
                try:
                    xr_file.vomecrty
                    self.attribute = xr_file.vomecrtry
                    self.velocity = 'V'
                except AttributeError:
                    raise AttributeError('No velocity attribute exists in netcdf file')
        self.begin_time = begin
        self.end_time = end
        self.__time_slice__()
        self.top_depth = top
        self.bottom_depth = bottom
        self.__depth_slice__()
        self.__slice__()
        self.attrs['Velocity'] = self.velocity
        self.attrs["Start_time"]=self.begin_time
        self.attrs["End_time"]=self.end_time
        self.attrs["Top_depth_layer"]=self.top_depth
        self.attrs["Bottom_depth_layer"]=self.bottom_depth
        self.attrs["Sea_floor_layer"]=self.sea_floor
        self.__plot__(cmap,title)
        if verbose is True:
            print(self.attrs)
        
    def __plot__(self,cmap, title):
        self.data.T.plot(cmap = cmap)
        lat, lon = round(float(self.xr_file.nav_lat),5), round(float(self.xr_file.nav_lon),5)
        if title is None:
            plt.title('Latitude: {}, Longitude: {}'.format(str(lat), str(lon)))
        else:
            plt.title = title
        plt.gca().invert_yaxis()
    
    def __find_bottom__(self, both=True):
        i=0
        for value in np.flip(self.attribute.isel(time_counter = 0).values):
            if value != 0:
                if both is False:
                    self.sea_floor = 39-i
                if both is True:
                    self.bottom_depth = 39-i 
                    self.sea_floor = 39-i
            else:
                i=i+1
    
    def __slice__(self):
        velocity = self.velocity
        if velocity is 'U':
            self.data = self.attribute.sel(
                time_counter = slice(self.begin_time, self.end_time)
            ).isel(
                depthu = slice(self.top_depth, self.bottom_depth)
            )

        elif velocity is "V":
            self.data = self.attribute.sel(
                time_counter = slice(self.begin_time, self.end_time)
            ).isel(
                depthv = slice(self.top_depth, self.bottom_depth)
            )
        else:
            self.data = self.attribute.sel(
                time_counter = slice(self.begin_time, self.end_time)
            ).isel(
                depthw = slice(self.top_depth, self.bottom_depth)
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