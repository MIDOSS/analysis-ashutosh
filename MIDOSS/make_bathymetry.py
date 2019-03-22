import numpy as np
import xarray as xr

grid = xr.open_dataset('https://salishsea.eos.ubc.ca/erddap/griddap/ubcSSn2DMeshMaskV17-02')
bathymetry = xr.open_dataset('AfterNEMOBathy201702.nc').Bathymetry.values[1:897, 1:397]

# mask out land values with -99
condlist = [bathymetry == 0, bathymetry != 0]
choicelist = [-99, bathymetry]
bathymetry = np.select(condlist, choicelist)


# open the boundaries
left = bathymetry[...,0]
condlist = [left == -99, left != -99]
choicelist = [-99, 0]
bathymetry[...,0] = np.select(condlist, choicelist)

right = bathymetry[...,-1]
condlist = [right == -99, right != -99]
choicelist = [-99, 0]
bathymetry[...,-1] = np.select(condlist, choicelist)

top = bathymetry[0]
condlist = [top == -99, top != -99]
choicelist = [-99, 0]
bathymetry[0] = np.select(condlist, choicelist)

bottom = bathymetry[-1]
condlist = [bottom == -99, bottom != -99]
choicelist = [-99, 0]
bathymetry[-1] = np.select(condlist, choicelist)


# get the latitudes
u = grid.gphiu.isel(time = 0).values
u_roll = np.roll(u,shift = -1, axis = 0)
unstaggered_u = ((u + u_roll)/2)[0:897, 1:398]


# get the longitudes
v = grid.glamv.isel(time = 0).values 
v_roll = np.roll(v,shift = -1, axis = 1)
unstaggered_v = ((v + v_roll)/2)[0:897, 0:397]

# write to the dat file
bathy = open('Bathymetry.dat', 'wt')

bathy.write('ILB_IUB : 1 896\n')
bathy.write('JLB_JUB : 1 396\n')
bathy.write('COORD_TIP : 4\n')
bathy.write(f'ORIGIN : {unstaggered_v[0][0]} {unstaggered_u[0][0]}\n')
bathy.write('GRID_ANGLE : 0\n')
bathy.write('LATITUDE : 0\n')
bathy.write('LONGITUDE: 0\n')
bathy.write('FILL_VALUE : -99\n')
#write lat/lons
bathy.write('<CornersXY>\n')
for i in range(897):
    for j in range(397):
        bathy.write(f'{unstaggered_v[i][j]} {unstaggered_u[i][j]}\n')
bathy.write('<BeginGridData2D>\n')
# write bathymetries
for i in range(896):
    for j in range(396):
        bathy.write(f'{bathymetry[i][j]}\n')
bathy.write('<EndGridData2D>\n')

bathy.close()