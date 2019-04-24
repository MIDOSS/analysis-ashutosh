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

"""Produce bilinear interpolation weighting matrix to regrid Wave Watch 3
"""

import numpy as np
import xarray as xr 
from salishsea_tools import geo_tools, utilities
from time import time
from numba import jit
import cProfile
import pstats

def produce_weighting_matrix(tgt_lats, tgt_lons, tgt_mask, src_lats, src_lons, src_mask):
    """Produce the weighting matrix for regridding from a source grid to a target grid.
       This will be quite time consuming and is only recommended if the weighting matrix
       will be used multiple times.
       For smaller and/or few time jobs, scipy's griddata is likely to serve you better.

    :arg tgt_lats: 2D array of latitude values for each grid point on the target grid
    :type numpy.ndarray

    :arg tgt_lons: 2D array of latitude values for each grid point on the target grid
    :type numpy.ndarray

    :arg tgt_mask: 2D array with 0 for land points and 1 for water points on target grid
    :type numpy.ndarray

    :arg src_lats: 2D array of latitude values for each grid point on the source grid
    :type numpy.ndarray

    :arg src_lons: 2D array of latitude values for each grid point on the source grid
    :type numpy.ndarray

    :arg src_mask: 2D array with 0 for land points and 1 for water points on source grid
    :type numpy.ndarray

    :returns weights: 1D array containing 4 floats, representing interpolation weights
    :type numpy.ndarray

    :returns index_j: 1D array containing 4 floats, representing source j indices
    :type numpy.ndarray

    :returns index_j: 1D array containing 4 floats, representing source i indices
    :type numpy.ndarray

    :returns integer: 1 if produced using iterations or 2 is produced using weighting matrix 
    :type int
    """
    # initialise gird with NaN everywhere
    tgt_weights = np.zeros([898, 398, 4])
    tgt_weights[:] = np.nan
    tgt_y_indices = np.zeros([898, 398, 4])
    tgt_y_indices[:] = np.nan
    tgt_x_indices = np.zeros([898, 398, 4])
    tgt_x_indices[:] = np.nan
    which_arr = np.zeros([898, 398])
    which_arr[:] = np.nan
    min_src_lat, max_src_lat = src_lats.min(), src_lats.max()
    min_src_lon, max_src_lon = src_lons.min(), src_lons.max()
    bar = utilities.statusbar('Loading...')
    for j in bar(range(898)):
        for i in range(398):
            if tgt_mask[j][i] == 0:
                continue
            tgt_lon = tgt_lons[j][i]
            if (tgt_lon < min_src_lon) or (tgt_lon > max_src_lon):
                continue
            tgt_lat = tgt_lats[j][i]
            if (tgt_lat < min_src_lat) or (tgt_lat > max_src_lat):
                continue
            else:
                result = _search_bounding_box_ww3(tgt_lat, tgt_lon, src_lats, src_lons, src_mask)
            if type(result) is tuple:
                weights, indices_j, indices_i, which = result
                tgt_weights[j][i] = weights
                tgt_y_indices[j][i] = indices_j
                tgt_x_indices[j][i] = indices_i
                which_arr[j][i] = which
            else:
                continue
    grid_x = np.arange(398)
    grid_y = np.arange(898)
    corners = np.arange(4) + 1
    u = xr.DataArray(tgt_weights, coords  = [grid_y, grid_x, corners], dims= ['grid_y', 'grid_x', 'index'])
    v = xr.DataArray(tgt_y_indices.astype(int), coords  = [grid_y, grid_x, corners], dims= ['grid_y', 'grid_x', 'index'])
    w = xr.DataArray(tgt_x_indices.astype(int), coords  = [grid_y, grid_x, corners], dims= ['grid_y', 'grid_x', 'index'])
    which_xr = xr.DataArray(which_arr, coords  = [grid_y, grid_x], dims= ['grid_y', 'grid_x'])
    a = xr.Dataset({'weights': u, 'y':v, 'x' : w, 'which': which_xr})
    a.to_netcdf('ww3_weighting_matrix_hakeV1.nc', format = 'NETCDF4',engine = 'netcdf4')

@jit
def _search_bounding_box_ww3(tgt_lat, tgt_lon, src_lats, src_lons, src_mask):
    """Search for a bounding box for a target lat-lon coordinate. 
       If none found, do a distance weighted average.
       Return the weights and corresponding array indices.
       This will make or break your weighting matrix.

    :arg tgt_lat: latitude of target grid point
    :type float

    :arg tgt_lon: longitude of target grid point
    :type float

    :arg src_lats: 2D array of latitude values for each grid point on the source grid
    :type numpy.ndarray

    :arg src_lons: 2D array of latitude values for each grid point on the source grid
    :type numpy.ndarray

    :arg src_mask: 2D array with 0 for land points and 1 for water points on source grid
    :type numpy.ndarray


    """
    boxes_found = 0
    box_attrs = []
    i_max, j_max = src_lons.shape[1], src_lats.shape[0]
    # focal point i,j is index of (1)
    for j in range(j_max-1):
        for i in range(i_max-1):
            # case 1: []
            vert1_i, vert1_j = i, j
            vert2_i, vert2_j = i+1, j
            vert3_i, vert3_j = i+1, j+1
            vert4_i, vert4_j = i, j+1

            if _all_vertex_on_water(
                vert1_i, vert1_j,
                vert2_i, vert2_j,
                vert3_i, vert3_j,
                vert4_i, vert4_j,
                src_mask
                ):    

                vert1_lat, vert1_lon = src_lats[vert1_j][vert1_i], src_lons[vert1_j][vert1_i]
                vert2_lat, vert2_lon = src_lats[vert2_j][vert2_i], src_lons[vert2_j][vert2_i]
                vert3_lat, vert3_lon = src_lats[vert3_j][vert3_i], src_lons[vert3_j][vert3_i]
                vert4_lat, vert4_lon = src_lats[vert4_j][vert4_i], src_lons[vert4_j][vert4_i]
                
                is_point_bounded = _check_bound(
                    tgt_lat, tgt_lon,
                    vert1_lat, vert1_lon,
                    vert2_lat, vert2_lon,
                    vert3_lat, vert3_lon,
                    vert4_lat, vert4_lon
                    )
            
                if is_point_bounded:
                    box_attrs = _add_box_attribute(
                        tgt_lon, tgt_lat,
                        vert1_i, vert2_i, vert3_i, vert4_i,
                        vert1_j, vert2_j, vert3_j, vert4_j,
                        vert1_lat, vert2_lat, vert3_lat, vert4_lat,
                        vert1_lon, vert2_lon, vert3_lon, vert4_lon,
                        src_mask, box_attrs
                        )
                    boxes_found = 1 +  boxes_found
            
            # case 2: //
            if i <= i_max-3:
                vert1_i, vert1_j = i, j
                vert2_i, vert2_j = i+1, j
                vert3_i, vert3_j = i+2, j+1
                vert4_i, vert4_j = i+1, j+1
            
                if _all_vertex_on_water(
                    vert1_i, vert1_j,
                    vert2_i, vert2_j,
                    vert3_i, vert3_j,
                    vert4_i, vert4_j,
                    src_mask
                    ):   

                    vert1_lat, vert1_lon = src_lats[vert1_j][vert1_i], src_lons[vert1_j][vert1_i]
                    vert2_lat, vert2_lon = src_lats[vert2_j][vert1_i], src_lons[vert1_j][vert2_i]
                    vert3_lat, vert3_lon = src_lats[vert3_j][vert1_i], src_lons[vert1_j][vert3_i]
                    vert4_lat, vert4_lon = src_lats[vert4_j][vert1_i], src_lons[vert1_j][vert4_i]

                    is_point_bounded = _check_bound(
                        tgt_lat, tgt_lon,
                        vert1_lat, vert1_lon,
                        vert2_lat, vert2_lon,
                        vert3_lat, vert3_lon,
                        vert4_lat, vert4_lon,
                        )
                
                    if is_point_bounded:
                        box_attrs = _add_box_attribute(
                            tgt_lon, tgt_lat,
                            vert1_i, vert2_i, vert3_i, vert4_i,
                            vert1_j, vert2_j, vert3_j, vert4_j,
                            vert1_lat, vert2_lat, vert3_lat, vert4_lat,
                            vert1_lon, vert2_lon, vert3_lon, vert4_lon,
                            src_mask, box_attrs
                            )
                        boxes_found = 1 +  boxes_found
            
            # case 3: \\
            if i != 0:
                vert1_i, vert1_j = i, j
                vert2_i, vert2_j = i+1, j
                vert3_i, vert3_j = i, j+1
                vert4_i, vert4_j = i-1, j+1

                if _all_vertex_on_water(
                    vert1_i, vert1_j,
                    vert2_i, vert2_j,
                    vert3_i, vert3_j,
                    vert4_i, vert4_j,
                    src_mask
                    ):
                
                    vert1_lat, vert1_lon = src_lats[vert1_j][vert1_i], src_lons[vert1_j][vert1_i]
                    vert2_lat, vert2_lon = src_lats[vert2_j][vert1_i], src_lons[vert1_j][vert2_i]
                    vert3_lat, vert3_lon = src_lats[vert3_j][vert1_i], src_lons[vert1_j][vert3_i]
                    vert4_lat, vert4_lon = src_lats[vert4_j][vert1_i], src_lons[vert1_j][vert4_i]
            
                    is_point_bounded = _check_bound(
                        tgt_lat, tgt_lon,
                        vert1_lat, vert1_lon,
                        vert2_lat, vert2_lon,
                        vert3_lat, vert3_lon,
                        vert4_lat, vert4_lon,
                        )
                
                    if is_point_bounded:
                        box_attrs = _add_box_attribute(
                            tgt_lon, tgt_lat,
                            vert1_i, vert2_i, vert3_i, vert4_i,
                            vert1_j, vert2_j, vert3_j, vert4_j,
                            vert1_lat, vert2_lat, vert3_lat, vert4_lat,
                            vert1_lon, vert2_lon, vert3_lon, vert4_lon,
                            src_mask, box_attrs
                            )
                        boxes_found = 1 + boxes_found
            # case 4: |//|
            if j <= j_max-3:
                vert1_i, vert1_j = i, j
                vert2_i, vert2_j = i+1, j+1
                vert3_i, vert3_j = i+1, j+2
                vert4_i, vert4_j = i, j+1

                if _all_vertex_on_water(
                    vert1_i, vert1_j,
                    vert2_i, vert2_j,
                    vert3_i, vert3_j,
                    vert4_i, vert4_j,
                    src_mask
                    ): 
                
                    vert1_lat, vert1_lon = src_lats[vert1_j][vert1_i], src_lons[vert1_j][vert1_i]
                    vert2_lat, vert2_lon = src_lats[vert2_j][vert1_i], src_lons[vert1_j][vert2_i]
                    vert3_lat, vert3_lon = src_lats[vert3_j][vert1_i], src_lons[vert1_j][vert3_i]
                    vert4_lat, vert4_lon = src_lats[vert4_j][vert1_i], src_lons[vert1_j][vert4_i]
                    
                    is_point_bounded = _check_bound(
                        tgt_lat, tgt_lon,
                        vert1_lat, vert1_lon,
                        vert2_lat, vert2_lon,
                        vert3_lat, vert3_lon,
                        vert4_lat, vert4_lon
                        )
                
                    if is_point_bounded:
                        box_attrs = _add_box_attribute(
                            tgt_lon, tgt_lat,
                            vert1_i, vert2_i, vert3_i, vert4_i,
                            vert1_j, vert2_j, vert3_j, vert4_j,
                            vert1_lat, vert2_lat, vert3_lat, vert4_lat,
                            vert1_lon, vert2_lon, vert3_lon, vert4_lon,
                            src_mask, box_attrs
                            )
                        boxes_found = 1 + boxes_found
            # case 5: |\\|
            if j <= j_max - 2:
                vert1_i, vert1_j = i, j
                vert2_i, vert2_j = i+1, j-1
                vert3_i, vert3_j = i+1, j
                vert4_i, vert4_j = i, j+1

                if _all_vertex_on_water(
                    vert1_i, vert1_j,
                    vert2_i, vert2_j,
                    vert3_i, vert3_j,
                    vert4_i, vert4_j,
                    src_mask
                    ):
                    
                    vert1_lat, vert1_lon = src_lats[vert1_j][vert1_i], src_lons[vert1_j][vert1_i]
                    vert2_lat, vert2_lon = src_lats[vert2_j][vert1_i], src_lons[vert1_j][vert2_i]
                    vert3_lat, vert3_lon = src_lats[vert3_j][vert1_i], src_lons[vert1_j][vert3_i]
                    vert4_lat, vert4_lon = src_lats[vert4_j][vert1_i], src_lons[vert1_j][vert4_i]
                    is_point_bounded = _check_bound(
                        tgt_lat, tgt_lon,
                        vert1_lat, vert1_lon,
                        vert2_lat, vert2_lon,
                        vert3_lat, vert3_lon,
                        vert4_lat, vert4_lon
                        )
                
                    if is_point_bounded:
                        box_attrs = _add_box_attribute(
                            tgt_lon, tgt_lat,
                            vert1_i, vert2_i, vert3_i, vert4_i,
                            vert1_j, vert2_j, vert3_j, vert4_j,
                            vert1_lat, vert2_lat, vert3_lat, vert4_lat,
                            vert1_lon, vert2_lon, vert3_lon, vert4_lon,
                            src_mask, box_attrs
                            )
                        boxes_found = 1 + boxes_found
    # finally, check how many boxes we found
    if boxes_found == 0:
        return _distance_avg(tgt_lat, tgt_lon, src_lats, src_lons, src_mask)
    else:
        print(boxes_found)
        # loop through and find the one with the lowest average distance to the points
        min_avg_dist = 1000000
        min_avg_dist_ind = 0
        for distance_ind, box in enumerate(box_attrs):
            distance = box[4].sum()/4
            if distance < min_avg_dist:
                min_avg_dist = distance
                min_avg_dist_ind = distance_ind
        # produce that best box
        index_i, index_j, latitudes, longitudes, dist = box_attrs[min_avg_dist_ind]
        weights =  _find_weights_bounding_box(tgt_lat, tgt_lon, latitudes, longitudes)
        return (weights, index_j, index_i, 1)

@jit(nopython = True)
def _all_vertex_on_water(
    vert1_i, vert1_j,
    vert2_i, vert2_j,
    vert3_i, vert3_j,
    vert4_i, vert4_j,
    src_mask
    ):
    """Return false if any of the vertices is on land, true otherwise
    """
    if (
        src_mask[vert1_j][vert1_i] == 0
        ) or (
            src_mask[vert2_j][vert2_i] == 0
            ) or (
                src_mask[vert3_j][vert3_i] == 0
                ) or(
                    src_mask[vert4_j][vert4_i] == 0
                    ):
        return False
    return True
    
@jit(nopython=True)
def _find_weights_bounding_box(tgt_lat, tgt_lon, latitudes, longitudes):
    iterations = 1000
    converge = 1e-10
    """ if we are inside a bounding box, we want to iterate and find the weights
    """
    lat1, lat2, lat3, lat4 = latitudes
    lon1, lon2, lon3, lon4 = longitudes
    dlat1 = lat2 - lat1
    dlat2 = lat4 - lat1
    dlat3 = lat3 - lat2 - dlat2

    dlon1 = lon2 - lon1
    dlon2 = lon4 - lon1
    dlon3 = lon3 - lon2 - dlon2

    i_iter = 0.5
    j_iter = 0.5

    for iteration in range(iterations):
        ddlat =  tgt_lat - lat1 - dlat1*i_iter - dlat2*j_iter - dlat3*i_iter*j_iter
        ddlon =  tgt_lon - lon1 - dlon1*i_iter - dlon2*j_iter - dlon3*i_iter*j_iter

        mat1 = dlat1 + dlat3*j_iter
        mat2 = dlat2 + dlat3*i_iter
        mat3 = dlon1 + dlon3*j_iter
        mat4 = dlon2 + dlon3*i_iter

        det = mat1*mat4 - mat2*mat3

        corr_i = (ddlat*mat4 - mat2*ddlon)/det  # correction to i
        corr_j = (ddlon*mat1 - ddlat*mat3)/det  # correction to j

        i_iter = i_iter + corr_i
        j_iter = j_iter + corr_j
        
        if (abs(corr_i) <= converge) and (abs(corr_j) <= converge):
            w1 = (1 - i_iter)*(1 - j_iter)
            w2 = i_iter*(1 - j_iter)
            w3 = i_iter*j_iter
            w4 = j_iter*(1 - i_iter)
            weights = np.array([w1,w2,w3,w4])
            return weights

@jit
def _distance_avg(tgt_lat, tgt_lon, src_lats, src_lons, src_mask):
    """Distance averaged weighting. Find distance between the target coordinates 
       and each of the source coordinates
    """
    how_close = 2

    distances = geo_tools.haversine(tgt_lon,tgt_lat,src_lons,src_lats)
    # make a copy and unravel
    dist_copy = distances.copy().ravel()
    dist_copy.sort()

    indices_j = np.zeros([4])
    indices_j[:] = np.nan

    indices_i = np.zeros([4])
    indices_i[:] = np.nan

    chosen_distances = np.zeros([4])
    chosen_distances[:] = np.nan

    points_found = 0
    for item in dist_copy:
        if points_found == 4:
            break
        if item > how_close:
            break
        where = np.where(distances == item)
        # ! what if more than one point is found? 
        ind_j, ind_i = where[0][0], where[1][0]
        if src_mask[ind_j][ind_i] == 1:
            indices_j[points_found] = ind_j
            indices_i[points_found] = ind_i
            chosen_distances[points_found] = item
            points_found = 1 + points_found # increment this by one
            
    # finally, produce the weights
    if points_found == 0:
        return False
    smol_boi = 1e-31 # small number to prevent division by zero
    weighting_divisor = 0
    for distance in  chosen_distances:
        weighting_divisor = weighting_divisor + 1/(distance + smol_boi)
    weights = []
    for neighbour in range(points_found):
        weight = (1/(chosen_distances[neighbour] + smol_boi)) / weighting_divisor
        weights.append(weight)
    # now that we got the weights and the indices, produce that as an array
    weight_arr = np.zeros([4])
    weight_arr[:] = np.nan

    for i,weight in enumerate(weights):
        weight_arr[i] = weight

    return (weight_arr, indices_j, indices_i, 2)

@jit  
def _add_box_attribute(
    tgt_lon, tgt_lat,
    vert1_i, vert2_i, vert3_i, vert4_i,
    vert1_j, vert2_j, vert3_j, vert4_j,
    vert1_lat, vert2_lat, vert3_lat, vert4_lat,
    vert1_lon, vert2_lon, vert3_lon, vert4_lon,
    src_mask, box_attrs
    ):
    index_i = (vert1_i, vert2_i, vert3_i, vert4_i)
    index_j = (vert1_j, vert2_j, vert3_j, vert4_j)
    latitudes = (vert1_lat, vert2_lat, vert3_lat, vert4_lat)
    longitudes = (vert1_lon, vert2_lon, vert3_lon, vert4_lon)
    distances = geo_tools.haversine(tgt_lon, tgt_lat, longitudes, latitudes)
    box_attr = (index_i, index_j, latitudes, longitudes, distances)
    box_attrs.append(box_attr)

    return box_attrs

@jit(nopython = True)
def _check_bound(
    tgt_lat, tgt_lon,
    vert1_lat, vert1_lon,
    vert2_lat, vert2_lon,
    vert3_lat, vert3_lon,
    vert4_lat, vert4_lon
    ):

    # check if the point is contained within the bounding box

    # point cannot be contained if lat is lower than bouding box bottom edges
    if (tgt_lat < vert2_lat) and (tgt_lat < vert1_lat):
        return False
    # point cannot be contained if lat is greater than bounding box top edges
    if (tgt_lat > vert3_lat) and (tgt_lat > vert4_lat):
        return False
    # point cannot be contained if lon is less than bounding box left edges
    if (tgt_lon < vert4_lon) and (tgt_lon < vert1_lon):
        return False
    # point cannot be contained is lon is greater than bounding box right edges
    if (tgt_lon > vert3_lon) and (tgt_lon > vert2_lon):
        return False
    # They pass the first tests. Now run a cross product test.
    vert_lats = (vert1_lat, vert2_lat, vert3_lat, vert4_lat, vert1_lat)
    vert_lons = (vert1_lon, vert2_lon, vert3_lon, vert4_lon, vert1_lon)

    for vertex in range(4):
        vec1_lat = vert_lats[vertex + 1] - vert_lats[vertex]
        vec1_lon = vert_lons[vertex + 1] - vert_lons[vertex]
        vec2_lat = tgt_lat - vert_lats[vertex]
        vec2_lon = tgt_lon - vert_lons[vertex]

        # if cross product is positive for any of these vectors, the point is not contained
        cross_product = vec1_lon*vec2_lat - vec2_lon*vec1_lat
        if cross_product < 0:
            return False

    # the point is within the bounding box
    return True

if __name__ == '__main__':
    src = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSf2DWaveFields30mV17-02')
    f = np.meshgrid(src.longitude.values - 360 ,src.latitude.values)
    src_lats, src_lons = f[1], f[0]

    smask = src.ucur.isel(time = 0).values * 0
    condlist = [smask == 0, smask!=0]
    choicelist = [1,0]
    src_mask = np.select(condlist,choicelist)

    tgt = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSnBathymetryV17-02')
    tgt_lats, tgt_lons = tgt.latitude.values, tgt.longitude.values
    tgt_mask = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSn2DMeshMaskV17-02').isel(time = 0).tmaskutil.values

    prof = cProfile.Profile()
    prof.run('produce_weighting_matrix(tgt_lats, tgt_lons, tgt_mask, src_lats, src_lons, src_mask)')
    prof.dump_stats('output.prof')

    stream = open('profile_regrid.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    
