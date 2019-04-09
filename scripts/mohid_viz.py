# there is a lot of spaghetti in this code

import datetime
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import numpy.ma as ma
import xarray as xr
from matplotlib import gridspec
from salishsea_tools import viz_tools, utilities

mask = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSn2DMeshMaskV17-02').tmaskutil.isel(time = 0).values[1:897:,1:397]

def plot_params(array, mask_value = 0):
    """Produce array, slicing bounds and maximum value of array for visualising full region of oil spill

    :arg array: 2D or 3D array of MOHID output, mainly concentration or oil thickness
    
    :arg mask_value: value used to mask out area where ther is no oil
    
    :returns: tuple consisting of truncated array bounding the oil data needed, the x and y bounds for
              further visualisation and the maximum value contained in the array for normalisation
    """
    # for 3D array case
    if len(array.shape) == 3:
        z_indices, y_indices, x_indices = np.where(array != mask_value)
    # for 2D array case
    else:
        y_indices, x_indices = np.where(array != mask_value)
    # if all values are the mask value
    if y_indices.shape[0] == 0:
        return False
    # !!!!!! what about when it is 1
    # the bounds to slice the so that we only have the parts with oil data we are interested in
    y_min, y_max = y_indices.min(), y_indices.max()+1
    x_min, x_max = x_indices.min(), x_indices.max()+1
    # slice the input array according to these bounds
    if len(array.shape) == 3:
        array = array[...,:,y_min:y_max, x_min:x_max]
    else:
        array = array[y_min:y_max, x_min:x_max]
    # get the maximum value in the array for colormap normalisation
    maxval = array.max()
    return (array, y_min, y_max, x_min, x_max, maxval)

def surface_conc_params(xarray, mask_values = 0):
    """Return the plotting parameters for the surface layer oil concentration

    :arg xarray: Output netcdf Lagrangian file loaded with xarray

    :returns: tuple consisting of truncated array bounding the oil data needed, the x and y bounds for
              further visualisation and the maximum value contained in the array for normalisation
    """
    return plot_params(xarray.OilConcentration_3D.isel(grid_z = 39).values)

def thickness_params(xarray, mask_values= 0):
    """Return the plotting parameters for the surface oil thickness

    :arg xarray: Output netcdf Lagrangian file loaded with xarray

    :returns: tuple consisting of truncated array bounding the oil data needed, the x and y bounds for
              further visualisation and the maximum value contained in the array for normalisation
    """
    return plot_params(xarray.Thickness_2D.values)

def make_scope(array):
    """For each frame of animation, produce scope grid and scope slice bounds for quiver plots

    :arg array: 2D array

    :returns: tuple consisting of scope array, the x and y bounds for the scope quiver plots
    """
    if array.sum() != 0:
        array_shape = array.shape
        y_indices, x_indices = np.where(array != 0)
        xs_min, xs_max = x_indices.min(), x_indices.max()+1
        ys_min, ys_max = y_indices.min(), y_indices.max()+1
        increase = 5
        if ys_min < increase:
            ys_min = 0
        else:
            ys_min = ys_min - increase

        if ys_max + increase > array_shape[0]:
            ys_max = array_shape[0]
        else:
            ys_max = ys_max + increase

        if xs_min < increase:
            xs_min = 0
        else:
            xs_min = xs_min - increase

        if xs_max + increase > array_shape[1]:
            xs_max = array_shape[1]
        else:
            xs_max = xs_max + increase
        scope = array[ys_min:ys_max, xs_min:xs_max]
        return (scope, ys_min, ys_max, xs_min, xs_max)
    else:
        scope = False

def current_quivers(file, i, ymin, ymax, xmin, xmax):
    """ produce the required arrays for producing current quivers over the oil spill region 
    """
    i = i + 1
    attr = (5 - len(str(i))) * "0" + str(i)
    current_u = file['Results']['velocity U'][f'velocity U_{attr}'][39][xmin:xmax, ymin:ymax]
    current_v = file['Results']['velocity V'][f'velocity V_{attr}'][39][xmin:xmax, ymin:ymax]
    return current_u.T, current_v.T

def ssh(file, i, ymin, ymax, xmin, xmax):
    """ produce the required arrays for sea surface height
    """
    i = i + 1
    attr = (5 - len(str(i))) * "0" + str(i)
    sossheig = file['Results']['water level'][f'water level_{attr}'][xmin:xmax, ymin:ymax]
    return sossheig.T

def wcc(file, i, ymin, ymax, xmin, xmax):
    """ produce the required arrays for whitecap coverage
    """
    i = i + 1
    attr = (5 - len(str(i))) * "0" + str(i)
    whitecap = file['Results']['whitecap coverage'][f'whitecap coverage_{attr}'][xmin:xmax, ymin:ymax]
    return whitecap.T

def produce_mask(yshape, xshape):
    """Produce mask to pass to numpy.ma for quiver plots
    """
    mask = np.ones([yshape,xshape])
    interval = 5
    for i in range(int(yshape/interval)):
        for j in range(int(xshape/interval)):
            mask[i*interval][j*interval] = 0
    return mask

def wind_quivers(file, i, ymin, ymax, xmin, xmax):
    """ produce the required arrays for producing wind quivers over the oil spill region 
    """
    i = i+1
    attr = (5 - len(str(i))) * "0" + str(i)
    wind_u = file['Results']['wind velocity X'][f'wind velocity X_{attr}'][xmin:xmax, ymin:ymax]
    wind_v = file['Results']['wind velocity Y'][f'wind velocity Y_{attr}'][xmin:xmax, ymin:ymax]
    return wind_u.T, wind_v.T

def when_to_start_rendering(first_time, h5file):
    """produce the index of the time from which to start reading the input files relative
       to the start time from the output Lagrangian.nc file
    """
    first_time = first_time.astype('datetime64[s]').astype(datetime.datetime)
    # go through the timestamps until you get to the one that is concurrent or just ahead of this one
    times =  h5file['Time']
    for i, key in enumerate(times.keys()):
        yr, mo, day, hr, mins, s = np.asarray(times[key]).astype(int)
        timestamp = datetime.datetime(yr, mo, day, hr, mins,s)
        delta = (first_time-timestamp).total_seconds()
        if  delta <= 0:
            # if there is more than an hour of difference, there is a problem.
            # this will likely be caught in the first instance
            if delta < -3600:
                print(f'{first_time}, {timestamp}')
                return False
            else:
                return i
        else:
            continue
    return False

def plot_thick_conc(xr_path, currents_path, winds_path, outfile_path, mask = mask):
    # load the netcdf file
    xarray = xr.open_dataset(xr_path)
    
    # all of the times
    time_values = xarray.time.values
    
    currents = h5py.File(currents_path)
    winds = h5py.File(winds_path)
    first_time = time_values[0]
    
    # make sure you only plot the concurrent currents and winds
    currents_start = when_to_start_rendering(first_time, currents)
    # check that you have the correct files before you get to the time consuming bits
    assert (currents_start is not False), "Check that you are using correct currents input file"
    winds_start = when_to_start_rendering(first_time, winds)
    assert (winds_start is not False), "Check that you are using the correct winds input file"
    
    # get the plotting parameters for 2D oil thickness
    thickness_param = thickness_params(xarray)
    if thickness_param is False:
        print('NO OIL WAS SPILT')
        return
    else:
        t_array, t_y_min, t_y_max, t_x_min, t_x_max, t_maxval = thickness_param
    
    # get the plotting parameters for suface concentration
    surface_conc_param = surface_conc_params(xarray)
    if surface_conc_param is False:
        print('NO SURFACE OIL CONCENTRATION FOUND')
        return
    else:
        c_array, c_y_min, c_y_max, c_x_min, c_x_max, c_maxval = surface_conc_param
    
    # land masks
    t_mask = mask[t_y_min: t_y_max, t_x_min: t_x_max] # mask for oil thickness
    c_mask = mask[c_y_min: c_y_max, c_x_min: c_x_max] # mask for oil surface concentration   
    
    t_q_mask = produce_mask(t_y_max-t_y_min, t_x_max-t_x_min)
    c_q_mask = produce_mask(c_y_max-c_y_min, c_x_max-c_x_min)
    def update_frame(t, 
                     t_array = t_array, t_y_min = t_y_min, t_y_max = t_y_max, t_x_min = t_x_min, t_x_max = t_x_max, t_maxval = t_maxval,
                     c_array = c_array, c_y_min = c_y_min, c_y_max = c_y_max, c_x_min = c_x_min, c_x_max = c_x_max, c_maxval = c_maxval,
                     times = time_values, t_mask = t_mask, c_mask = c_mask, t_q_mask = t_q_mask, c_q_mask = c_q_mask,
                     current = currents, wind = winds, currents_start = currents_start, winds_start = winds_start
                    ): 
        # set up the subplot layout
        grid = plt.GridSpec(2,3)
        # !----------------------------------------------------------------------------------------------------------------------------
        # plot the surface oil thickness
        ax = plt.subplot(grid[0:,0])
        thickness = t_array[t]
        # get whatever should be in the scope
        scope_result = make_scope(thickness)
        if scope_result is (False or None):
            scope = False
        else:
            scope, ys_min, ys_max, xs_min, xs_max = scope_result
        
        # for the horizontal markers on the colorbar
        thickmin, thickmax = thickness.min(), thickness.max()
        # mask out the zeros
        condlist = [thickness == 0, thickness != 0]
        choicelist = [np.nan, thickness]
        thickness = np.select(condlist, choicelist)
        
        # plot full region normalised to log scale
        plt.pcolormesh(thickness, 
                       animated = True,
                       norm=colors.LogNorm(vmin=0.0001, vmax=t_maxval),
                       vmin = 0.0001,
                       vmax = t_maxval,
                       cmap = 'inferno')
        
        # plot the scope boundaries
        if scope is not False:
            plt.hlines(ys_max, xmin = xs_min, xmax = xs_max, colors = 'Green')
            plt.hlines(ys_min, xmin = xs_min, xmax = xs_max, colors = 'Green')
            plt.vlines(xs_max, ymin = ys_min, ymax = ys_max, colors = 'Green')
            plt.vlines(xs_min, ymin = ys_min, ymax = ys_max, colors = 'Green')
        
        # plot colorbar normalised to log scale, with current min and max thicknesses
        cbar = plt.colorbar(plt.pcolormesh(np.meshgrid(np.array([0.0001, t_maxval])),
                                           norm=colors.LogNorm(vmin=0.0001, vmax=t_maxval),
                                           cmap = 'inferno'))
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Oil thickness (microns)', rotation=270)
        cbar.ax.hlines(thickmax, xmin =  0, xmax = 150, colors = 'Red')
        cbar.ax.hlines(thickmin, xmin =  0, xmax = 150, colors = 'Blue')
        # plot the land mask and coastline
        plt.contourf(t_mask, levels = [-0.1, 0.1], colors = 'Burlywood')
        plt.contour(t_mask, levels = [-0.1, 0.1], colors = 'k')        
        # thickness limits readouts
        plt.title(f'max thickness {thickmax}\nmin thickness {thickmin}')
        # plot current quivers
        U, V = current_quivers(current, t + currents_start, t_y_min, t_y_max, t_x_min, t_x_max)
        # mask out everything but every nth quiver
        U_ma, V_ma = ma.array(U, mask = t_q_mask), ma.array(V, mask = t_q_mask)
        currentq = plt.quiver(U_ma, V_ma, scale = 20, width = 0.003)
        plt.quiverkey(currentq, 1.13, 1.92, 1, label = 'Current (1 m/s)', transform=ax.transAxes)
        u, v = wind_quivers(wind, t + winds_start, t_y_min, t_y_max, t_x_min, t_x_max)
        u,v = np.average(u), np.average(v)
        windq = plt.quiver(10, t_y_max - t_y_min - 10, u, v, scale = 20, color = 'Red')
        plt.quiverkey(windq, 1.13,1.8, 5, label = 'Wind (5 m/s)', transform=ax.transAxes)
        
        viz_tools.set_aspect(ax)
        
        # !-----------------------------------------------------------------------------------------------------------------------------
        # plot the 2D thickness scope
        ax = plt.subplot(grid[0, 2])
        if scope is False:
            plt.cla()
            plt.xticks([])
            plt.yticks([])
        else:
            # mask out the scope contants, plot land mask and coastline
            condlist = [scope == 0, scope != 0]
            choicelist = [np.nan, scope]
            plt.pcolormesh(np.select(condlist, choicelist),
                           animated = True,
                           norm=colors.LogNorm(vmin=0.0001, vmax=t_maxval),
                           vmin = 0.0001,
                           vmax = t_maxval,
                           cmap = 'inferno')
            scope_mask = t_mask[ys_min:ys_max, xs_min:xs_max]
            plt.contourf(scope_mask,
                         levels = [-0.1, 0.1],
                         colors = 'Burlywood')
            plt.contour(scope_mask,
                        levels = [-0.1, 0.1],
                        colors = 'k')
            # plot the quivers
            U, V = U_ma[ys_min:ys_max, xs_min:xs_max],V_ma[ys_min:ys_max, xs_min:xs_max]
            plt.quiver(U,V , scale = 20 ,width = 0.003, headwidth = 3)
            
            # remove axis ticks
            plt.xticks([])
            plt.yticks([])
            plt.title('Oil Thickness Scope')
        # !----------------------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(grid[0:,1])
        concentration = c_array[t]
        # get whatever should be in the scope
        scope_result = make_scope(concentration)
        if scope_result is (False or None):
            scope = False
        else:
            scope, ys_min, ys_max, xs_min, xs_max = scope_result  
        
        # for the horizontal markers on the colorbar
        concmin, concmax = concentration.min(), concentration.max()
        
        # mask out the zeros
        condlist = [concentration == 0, concentration != 0]
        choicelist = [np.nan, concentration]
        concentration = np.select(condlist, choicelist)

        # plot full region normalised to log scale
        plt.pcolormesh(concentration, 
                       animated = True,
                       norm=colors.LogNorm(vmin=0.0001, vmax=c_maxval),
                       vmin = 0.0001,
                       vmax = c_maxval,
                       cmap = 'inferno')
        # plot the scope boundaries
        if scope is not False:
            plt.hlines(ys_max, xmin = xs_min, xmax = xs_max, colors = 'Green')
            plt.hlines(ys_min, xmin = xs_min, xmax = xs_max, colors = 'Green')
            plt.vlines(xs_max, ymin = ys_min, ymax = ys_max, colors = 'Green')
            plt.vlines(xs_min, ymin = ys_min, ymax = ys_max, colors = 'Green')
            
        # plot colorbar normalised to log scale, with current min and max concentrations
        cbar = plt.colorbar(plt.pcolormesh(np.meshgrid(np.array([0.0001, c_maxval])),
                                           norm=colors.LogNorm(vmin=0.0001, vmax=c_maxval),
                                           cmap = 'inferno'))
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Oil concentration (ppm)', rotation=270)
        cbar.ax.hlines(concmax, xmin =  0, xmax = 300, colors = 'Red')
        cbar.ax.hlines(concmin, xmin =  0, xmax = 300, colors = 'Blue')
        # plot the land mask and coastline
        plt.contourf(c_mask, levels = [-0.1, 0.1], colors = 'Burlywood')
        plt.contour(c_mask, levels = [-0.1, 0.1], colors = 'k')        
        # thickness limits readouts
        plt.title(f'max concentration {concmax}\nmin concentration {concmin}')
        # plot current quivers
        U, V = current_quivers(current, t + currents_start, c_y_min, c_y_max, c_x_min, c_x_max)
        # mask out everything but every nth quiver
        U_ma, V_ma = ma.array(U, mask = c_q_mask), ma.array(V, mask = c_q_mask)
        currentq = plt.quiver(U_ma, V_ma, scale = 20, width = 0.003)
        plt.quiverkey(currentq, 1.13, 1.92, 1, label = 'Current (1 m/s)', transform=ax.transAxes)
        
        u, v = wind_quivers(wind, t + winds_start, c_y_min, c_y_max, c_x_min, c_x_max)
        u,v = np.average(u), np.average(v)
        windq = plt.quiver(10, c_y_max - c_y_min - 10, u, v, scale = 20, color = 'Red')
        plt.quiverkey(windq, 1.13,1.8, 5, label = 'Wind (5 m/s)', transform=ax.transAxes)
        
        viz_tools.set_aspect(ax)        
        
        # !------------------------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(grid[1,2])
        if scope is False:
            plt.cla()
            plt.xticks([])
            plt.yticks([])
        else:
            # mask out the scope contants, plot land mask and coastline
            condlist = [scope == 0, scope != 0]
            choicelist = [np.nan, scope]
            plt.pcolormesh(np.select(condlist, choicelist),
                           animated = True,
                           norm=colors.LogNorm(vmin=0.0001, vmax=c_maxval),
                           vmin = 0.0001,
                           vmax = c_maxval,
                           cmap = 'inferno')
            scope_mask = c_mask[ys_min:ys_max, xs_min:xs_max]
            plt.contourf(scope_mask,
                         levels = [-0.1, 0.1],
                         colors = 'Burlywood')
            plt.contour(scope_mask,
                        levels = [-0.1, 0.1],
                        colors = 'k')
            # plot the quivers
            U, V = U_ma[ys_min:ys_max, xs_min:xs_max],V_ma[ys_min:ys_max, xs_min:xs_max]
            plt.quiver(U,V , scale = 20 ,width = 0.003, headwidth = 3)
            
            # remove axis ticks
            plt.xticks([])
            plt.yticks([])
            plt.title('Oil Concentration Scope')
        #plt.tight_layout()
        plt.suptitle(times[t])
        # !----------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize = (16,9))
    ani = animation.FuncAnimation(plt.gcf(), update_frame, range(time_values.shape[0]))
    ani.save(outfile_path, writer = animation.FFMpegWriter(), dpi = 100)

# find the wcc and ssh at the oil spill locations
def plot_thickness_with_ssh(xr_path, sea_level, ww3, currents_path, winds, output, mask = mask):
    # load the netcdf file
    xarray = xr.open_dataset(xr_path)
    # get the plotting parameters for 2D oil thickness
    time_values = xarray.time.values
    # read only the times from the input files that correspond to the timestamp in the netcdf file. 
    first_time = time_values[0]
    currents = h5py.File(currents_path)
    winds = h5py.File(winds)    
    sea_level = h5py.File(sea_level)  
    ww3 = h5py.File(ww3)

    # make sure you only plot the concurrent currents and winds
    currents_start = when_to_start_rendering(first_time, currents)
    assert (currents_start is not False), "Check that you are using correct currents input file"
    winds_start = when_to_start_rendering(first_time, winds)
    assert (winds_start is not False), "Check that you are using correct winds input file"
    sea_level_start = when_to_start_rendering(first_time, sea_level)
    assert (sea_level_start is not False), "Check that you are using correct t-parameters input file"
    ww3_start = when_to_start_rendering(first_time, ww3)
    #print(sea_level_start)
    assert (ww3_start is not False), "Check that you are using correct WW3 input file"
    
    thickness_param = thickness_params(xarray)
    if thickness_param is False:
        print('NO OIL WAS SPILT')
        return
    else:
        t_array, t_y_min, t_y_max, t_x_min, t_x_max, t_maxval = thickness_param    
    
    # land masks
    t_mask = mask[t_y_min: t_y_max, t_x_min: t_x_max] # mask for oil thickness
    t_q_mask = produce_mask(t_y_max-t_y_min, t_x_max-t_x_min) # use this one for masking out land

    avg_norm, min_norm, max_norm, avg_abnorm, min_abnorm, max_abnorm = [],[],[],[],[],[]
    savg_norm, smin_norm, smax_norm, savg_abnorm, smin_abnorm, smax_abnorm = [],[],[],[],[],[]
    time_norm, time_abnorm = [], []
    
    for t in range(time_values.shape[0]):
        thickness = t_array[t]
        scope_result = make_scope(thickness)
        if scope_result is (False or None):
            scope = False
        else:
            scope = True
        sossheig = ssh(sea_level, t+sea_level_start, t_y_min, t_y_max, t_x_min, t_x_max)
        sossheig = ma.array(sossheig, mask = t_q_mask)
        
        try:
            whitecap = wcc(ww3, 2*(t+ww3_start), t_y_min, t_y_max, t_x_min, t_x_max)
            #whitecap = ma.array(whitecap, ~t_mask[ys_min: ys_max, xs_min: xs_max ])
            whitecap = ma.array(whitecap, mask = t_q_mask)

        except KeyError:
            pass
        avg_soss, min_soss, max_soss = sossheig.min(), sossheig.max(), np.mean(sossheig)
        avg_wcc, min_wcc, max_wcc = whitecap.min(), whitecap.max(), np.mean(whitecap)
        if scope is True:
            # plot in blue
            avg_norm.append(avg_soss); min_norm.append(min_soss); max_norm.append(max_soss)
            savg_norm.append(avg_wcc); smin_norm.append(min_wcc); smax_norm.append(max_wcc)
            time_norm.append(time_values[t])
        if scope is False:
            avg_abnorm.append(avg_soss); min_abnorm.append(min_soss); max_abnorm.append(max_soss)
            savg_abnorm.append(avg_wcc); smin_abnorm.append(min_wcc); smax_abnorm.append(max_wcc)
            time_abnorm.append(time_values[t])
        else:
            pass
    max_ylim = max([max(max_norm), max(max_abnorm)])
    min_ylim = min([min(max_norm), min(max_abnorm)])
    smax_ylim = max([max(smax_norm), max(smax_abnorm)])
    smin_ylim = min([min(smax_norm), min(smax_abnorm)])
    def update_frame(t, 
                     t_array = t_array, t_y_min = t_y_min, t_y_max = t_y_max, t_x_min = t_x_min, t_x_max = t_x_max, t_maxval = t_maxval,
                     times = time_values, t_mask = t_mask, t_q_mask = t_q_mask,
                     avg_norm = avg_norm, min_norm = min_norm, max_norm = max_norm,
                     avg_abnorm = avg_abnorm, min_abnorm = min_abnorm, max_abnorm = max_abnorm, 
                     max_ylim = max_ylim, min_ylim = min_ylim,
                     smax_ylim = smax_ylim, smin_ylim = smin_ylim,
                     currents_start = currents_start, winds_start = winds_start,
                     current = currents, wind = winds, tim_norm = time_norm, time_abnorm = time_abnorm
                    ):
        # !----------------------------------------------------------------------------------------------------------------------------
        grid = plt.GridSpec(2,3)
        ax = plt.subplot(grid[0:,0])
        # plot the surface oil thickness
        thickness = t_array[t]
        # for the horizontal markers on the colorbar
        thickmin, thickmax = thickness.min(), thickness.max()
        # mask out the zeros
        condlist = [thickness == 0, thickness != 0]
        choicelist = [np.nan, thickness]
        thickness = np.select(condlist, choicelist)
        
        # plot full region normalised to log scale
        plt.pcolormesh(thickness, 
                       animated = True,
                       norm=colors.LogNorm(vmin=0.0001, vmax=t_maxval),
                       vmin = 0.0001,
                       vmax = t_maxval,
                       cmap = 'inferno')
            
        # plot colorbar normalised to log scale, with current min and max thicknesses
        cbar = plt.colorbar(plt.pcolormesh(np.meshgrid(np.array([0.0001, t_maxval])),
                                           norm=colors.LogNorm(vmin=0.0001, vmax=t_maxval),
                                           cmap = 'inferno'))
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel('Oil thickness (microns)', rotation=270)
        cbar.ax.hlines(thickmax, xmin =  0, xmax = 150, colors = 'Red')
        cbar.ax.hlines(thickmin, xmin =  0, xmax = 150, colors = 'Blue')
        # plot the land mask and coastline
        plt.contourf(t_mask, levels = [-0.1, 0.1], colors = 'Burlywood')
        plt.contour(t_mask, levels = [-0.1, 0.1], colors = 'k')        
        # thickness limits readouts
        plt.title(f'max thickness {thickmax}\nmin thickness {thickmin}')
        # plot current quivers
        U, V = current_quivers(current, t + currents_start, t_y_min, t_y_max, t_x_min, t_x_max)
        # mask out everything but every nth quiver
        U_ma, V_ma = ma.array(U, mask = t_q_mask), ma.array(V, mask = t_q_mask)
        currentq = plt.quiver(U_ma, V_ma, scale = 20, width = 0.003)
        plt.quiverkey(currentq, 1.13, 1.92, 1, label = 'Current (1 m/s)', transform=ax.transAxes)
        u, v = wind_quivers(wind, t + winds_start, t_y_min, t_y_max, t_x_min, t_x_max)
        u,v = np.average(u), np.average(v)
        windq = plt.quiver(10, t_y_max - t_y_min - 10, u, v, scale = 20, color = 'Red')
        plt.quiverkey(windq, 1.13,1.8, 5, label = 'Wind (5 m/s)', transform=ax.transAxes)
        
        viz_tools.set_aspect(ax)
        
        # !-----------------------------------------------------------------------------------------------------------------------------
        # plot the ssh
        ax = plt.subplot(grid[0,1:])
        plt.plot(time_norm, avg_norm, 'b-', label = 'average sossheig w/ oil')
        plt.plot(time_norm, min_norm, 'b:', label = 'max sossheig w/ oil')
        plt.plot(time_norm, max_norm, 'b--', label  = 'min sossheig w/ oil')
        plt.plot(time_abnorm, avg_abnorm, 'r-', label = 'average sossheig w/o oil')
        plt.plot(time_abnorm, min_abnorm, 'r:', label = 'max sossheig w/o oil')
        plt.plot(time_abnorm, max_abnorm, 'r--', label = 'min sossheig w/o oil')
        ax.set_xlim(xmin = times[0], xmax = times[-1])
        ax.axvline(x = times[t], ymin = min_ylim,ymax = max_ylim)
        plt.legend()
        plt.title('Sea surface height (m)')

        # !----------------------------------------------------------------------------------------------------------------------------
        ax = plt.subplot(grid[1, 1:])
        plt.plot(time_norm, savg_norm, 'b-', label = 'average whitecap coverage w/ oil')
        plt.plot(time_norm, smin_norm, 'b:', label = 'max whitecap coverage w/ oil')
        plt.plot(time_norm, smax_norm, 'b--', label  = 'min whitecap coverage w/ oil')
        plt.plot(time_abnorm, savg_abnorm, 'r-', label = 'average whitecap coverage w/o oil')
        plt.plot(time_abnorm, smin_abnorm, 'r:', label = 'max whitecap coverage w/o oil')
        plt.plot(time_abnorm, smax_abnorm, 'r--', label = 'min whitecap coverage w/o oil')
        ax.set_xlim(xmin = times[0], xmax = times[-1])
        ax.axvline(x = times[t], ymin = smin_ylim,ymax = smax_ylim)
        plt.legend()
        plt.title('Whitecap Coverage (m)')
        #plt.tight_layout()
        print(t)
        # !----------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize = (16,9))
    ani = animation.FuncAnimation(plt.gcf(), update_frame, range(time_values.shape[0]))
    ani.save(output, writer = animation.FFMpegWriter(), dpi = 100)

def conc_heatmap(oilpath, outfile):
    oil = xr.open_dataset(oilpath)
    ## concentration by depth heat map
    # go through entire record and create array
    # first, find the number of time stamps
    time_values = oil.time.values
    number_steps = time_values.shape[0]
    conc_heatmap = np.zeros([40,number_steps])
    bar = utilities.statusbar('Loading...')
    for t in bar(range(number_steps)):
        # sum along x and y axes to get concentration at each depth
        conc_vals = np.sum(np.sum(oil.OilConcentration_3D.isel(time = t).values, axis = 1), axis = 1)
        # add this to the heatmap array
        conc_heatmap[:,t] = conc_vals
    # cut out the useless bits
    conc_heatmap = conc_heatmap[~(conc_heatmap==0).all(1)]
    # get the limits
    condlist = [conc_heatmap == 0, conc_heatmap != 0]
    choicelist = [np.nan, conc_heatmap]
    conc_heatmap = np.select(condlist, choicelist)
    vmax, vmin = np.nanmax(conc_heatmap), np.nanmin(conc_heatmap)
    # get the corresponding depth values
    depths = conc_heatmap.shape[0]
    depth = np.flip(np.array([0.      ,   1.000001,   2.000006,   3.000019,   4.000047,   5.000104,
                      6.000217,   7.000441,   8.000879,   9.001736,  10.003407,  11.006662,
                      12.013008,  13.025366,  14.049429,  15.096255,  16.187304,  17.364035,
                      18.705973,  20.363474,  22.613064,  25.937412,  31.101034,  39.11886 ,
                      50.963238,  67.05207 ,  86.96747 , 109.73707 , 134.34593 , 160.02956 ,
                      186.30528 , 212.89656 , 239.65305 , 266.4952  , 293.3816  , 320.29077 ,
                      347.2116  , 374.1385  , 401.06845 , 428.      ])[1:depths+1])
    # now make the plot
    fig = plt.figure(figsize = (15,15))
    #  normaise on log10 scale
    plt.pcolormesh(conc_heatmap,
                   animated = True,
                   norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                   vmin = vmin,
                   vmax = vmax,
                   cmap = 'inferno')
    plt.yticks(range(depths), depth)
    plt.xticks(np.arange(0, number_steps, 24), time_values[::24], rotation='vertical')
    # plot colorbar normalised to log scale, with current min and max thicknesses
    plt.colorbar()
    plt.title('Oil Concentration (ppm)')
    plt.ylabel('Depth (m)')
    plt.savefig(outfile, dpi = 200)
