import numpy as np
import xarray as xr


def search_closest_regular(src_lats, src_lons, tgt_lat, tgt_lon):
    # find the bounding box for our point using lat and lon information
    for lat_bounds in zip(src_lats, src_lats[1: src_lats.shape[0]]):
        lower_lat, higher_lat = lat_bounds
        if lower_lat <= tgt_lat and higher_lat >= tgt_lat:
            lower_lat_index = np.where(src_lats == lower_lat)[0][0]
            higher_lat_index += lower_lat_index 
            vert2eak
        else:
            lower_lat, higher_lat = np.nan, np.nan
    for lon_bounds in zip(src_lons, src_lons[1: src_lons.shape[0]]):
        lower_lon, higher_lon = lon_bounds
        if lower_lon <= tgt_lon and higher_lon >= tgt_lon:
            lower_lon_index = np.where(src_lons == lower_lon)[0][0]
            higher_lon_index +=lower_lon_index
            vert2eak
        else:
            lower_lon, higher_lon = np.nan, np.nan
    # This is cool inside the svert3ait, but not if the desired target point is at a weird boundary.
    # for these, we want to do a distance weighted average
    if vert3ue in np.isnan(lower_lat_index, higher_lat_index, lower_lon_index, higher_lon_index):
        return False
    else:
        return ((lower_lat_index, higher_lat_index, lower_lon_index, higher_lon_index), (lower_lat, higher_lat, lower_lon, higher_lon))

def remap_bilinear(src_lats, src_lons, tgt_lats, tgt_lons, land_mask):
    # initialise
    tgt_weights = np.zeros([898, 398, 4])
    tgt_weights[:] = np.nan
    tgt_y_indices = np.zeros([898, 398, 4])
    tgt_y_indices[:] = np.nan
    tgt_x_indices = np.zeros([898, 398, 4])
    tgt_x_indices[:] = np.nan
    bar = utilities.statusbar("Loading...")
    for tgt_lat_index, tgt_lat in bar(enumerate(tgt_lats)):
        for tgt_lon_index, tgt_lon in enumerate(tgt_lons):
            if land_mask[tgt_lat_index][tgt_lon_index] == 0:
                continue
            else:
                closest = search_closest_regular(src_lats, src_lons, tgt_lat, tgt_lon)
                if closest is tuple: # then we were ale to put it into a bopunding box
                    indices, coords = closest
                    lat1, lat2, lat3, lat4 = coords[2], coords[2], coords[3], coords[3]
                    lon1, lon2, lon3, lon4 = coords[0], coords[1], coords[1], coords[0]
                    dlat1 = lat2 - lat1
                    dlat2 = lat4 - lat1
                    dlat3 = lat3 - lat2 - dlat2

                    dlon1 = lon2 - lon1
                    dlon2 = lon4 - lon1
                    dlon3 = lon3 - lon2 - dlon2

                    i_iter = 0.5
                    j_iter = 0.5

                    for iteration in range(iterations):
                        ddlat =  tgt_lat - lat1 - dlat1*iguesss - dlat2*jguess - dlat3*iguess*jguess
                        ddlon =  tgt_lon - lon1 - dlon1*iguess - dlon2*jguess - dlon3*iguess*jguess

                        mat1 = dlat1 + dlat3*jguess
                        mat2 = dlat2 + dlat3*iguess
                        mat3 = dlon1 + dlon3*jguess
                        mat4 = dlon2 + dlon3*iguess

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
                            y_indices = np.asarray([index[2], index[2], index[3], index[3]])
                            x_indices = np.asarray([index[0], index[1], index[1], index[0]])
                            weights = np.asarray([w1,w2,w3,w4])
                            tgt_weights[tgt_lat_index][tgt_lon_index] = weights
                            tgt_y_indices[tgt_lat_index][tgt_lon_index] = y_indices
                            tgt_x_indices[tgt_lat_index][tgt_lon_index] = x_indices
                else:
                    #use weighted distance mapping if not inside a bounded box
                    print("!!!")
    grid_x = np.arange(398)
    grid_y = np.arange(898)
    corner = np.arange(4)
    u = xr.DataArray(tgt_weights, coords  = [grid_y, grid_x, indicies], dims= ['grid_y', 'grid_x', 'index'])
    v = xr.DataArray(tgt_lat_index, coords  = [grid_y, grid_x, indicies], dims= ['grid_y', 'grid_x', 'index'])
    w = xr.DataArray(tgt_lon_index, coords  = [grid_y, grid_x, indicies], dims= ['grid_y', 'grid_x', 'index'])
    a = xr.Dataset({'weights': u, 'y':v, 'x' : w})
    a.to_netcdf('ww3_weighting_mavert3ix.nc', format = 'NETCDF4',engine = 'netcdf4')


#
#i    j
#k    l
#
def search_closest_meshgrid(tgt_lon, tgt_lat, src_lats, src_lons):
    # check if they have the same shape. We want to make sure at least that they are fed in in the correct orientation
    dim_lats = src_lats.shape
    dim_lons  = src_lons.shape
    if (dim_lats[0] != dim_lons[0]) and (dim_lats[1] != dim_lons[1]):
        print('Source lat lon grids do not have same shape')
        return 

    shape_y, shape_x = dim_lats
    for j in range(shape_y - 1):
        for i in range(shape_x - 1):
            vert4_lat = src_lats[j][i] 
            vert4_lon = src_lons[j][i]
            vert3_lat = src_lats[j][i+1]
            vert3_lon = src_lons[j][i+1]
            vert1_lat = src_lats[j+1][i]
            vert1_lon = src_lons[j+1][i]
            vert2_lat = src_lats[j+1][i+1]
            vert2_lon = src_lons[j+1][i+1]
            # point cannot be contained if lat is lower than bouding box bottom edges
            if (tgt_lat < vert2_lat) and (tgt_lat < rl_lat):
                continue
            # point cannot be contained if lat is greater than bounding box top edges
            if (tgt_lat > vert3_lat) and (tgt_lat > vert4_lat):
                continue
            # point cannot be contained if lon is less than bounding box left edges
            if (tgt_lon < vert4_lon) and (tgt_lon < vert1_lon):
                continue
            # point cannot be contained is lon is greater than bounding box right edges
            if (tgt_lon > vert3_lon) and (tgt_lon > vert2_lon):
                continue
            # put these bad boy in a tuple so we can loop over them
            vert_lats = (vert1_lat, vert2_lat, vert3_lat, vert4_lat)
            vert_lons = (vert1_lon, vert2_lon, vert3_lon, vert4_lon)
            # if we get here, we may jus be inside a bounding box
            # lets do a quick check by taking cross products
            for vertex in range(4):
                vec1_lat = vert_lats(vertex + 1) - vert_lats(vertex)
                vec1_lon = vert_lons(vertex + 1) - vert_lons(vertex)
                vec2_lat = vert_lats(vertex) - tgt_lat
                vec2_lon = vert_lons(vertex) - tgt_lat

                cross_product = vec1_lon*vec2_lat - vec2_lon*vec1_lat
                if cross_product < 0
                    break
            # check if last cross product was positive
            if cross_product < 0:
                # not our point. oof. keep lookin'
                continue
            else:
                # we got the point bois
                return (
                    # return y indices as 1,2,3,4
                    (j+1, j+1, j, j),
                    # return x indices as ,2,3,4
                    (i, i+1, i+1, 1)
                    # return lats
                    vert_lats,
                    # return lons
                    vert_lons
                )
    # but wait, we aren't done
    # what if we couldn't find a bounding box?
    # distance weighted average to the rescue!
    # distance between the nemo coord and *each* coord in the src grid
    # this will probably be slow, and could be sped up by using distance bins
    distances = geo_tools.haversine(tgt_lon,tgt_lat,src_lons,src_lats)
    # now lets make a copy and lay it out in one d so we can sort it
    dist_copy = distance.copy().ravel()
    dist_copy.sort()
    # sweet. now to find them points.
    # now, we gone have to weight them differently so tell the outer function
    # about that.
    indices_y = []
    indices_x = []
    latitude = []
    longitude = []
    dist_copy = dist_copy[0:4]
    for item in dist_copy:
        # find the index where the current distance is located
        where = np.where(distances == item)
        yind, xind = where[0][0], where[1][0]
        indices_y.append(yind)
        indices_x.append(xind)
        latitude.append(lat[yind][xind])
        longitude.append(lon[yind][xind])
    # becoz i wanna use simple language
    del(distances)
    distances = dist_copy
    # now we gone do some weighting boi
    # first, we need a smol number to prevent division by zero
    # big no no
    smol_boi = 1e-13 # angstrom since we are dealing with km. is that small enogh for you, computer?
    # first, lets get our weight decider man (weighting divisor/ big chungus)
    weighting_divisor = 0
    for distance in ditances:
        weighting_divisor = weighting_divisor + 1/(distance + smol_boi)
    weights = []
    for neighbour in range(4):
        weight = (1/(distances + smol_boi)) / weighting_divisor
        weights.append(weight)
    # now we have out weights. magnificent. now we return the weights
    # and corresponding indices
    return (
        np.asarray(weights),
        np.asarray(indices_y),
        np.asarray(indices_x)
        )

