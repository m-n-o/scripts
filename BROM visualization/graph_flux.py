#  The script is used for calculating fluxes from an only z-level

import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter.filedialog import askopenfilenames, askopenfilename

dims = ['i']  # dimensions to integrate over (i, and time)
flag = 'one'  # 'one' or 'two', only experimental or experimental with base values on one graph
first = '2012-01-06'  # df['time'].min()  # start time 'YYYY-MM-DD'
last = '2012-01-31'  # df['time'].max()  # end time 'YYYY-MM-DD'

flux_vars = ['MgOH2', 'MgOH2_diff',
             'pCO2', 'pCO2_diff',
             'CaCO3', 'CaCO3_diff',
             'Alk', 'Alk_diff',
             'DIC', 'DIC_diff']  # variable to integrate flux ('pCO2' or 'pCO2_diff')


def get_exp_fname():
    """
    Function that call the window for .nc file selection
    """
    fname = askopenfilenames(
        initialdir=os.getcwd(),
        filetypes=(("netcdf file", "*.nc"), ("All Files", "*.*")),
        title="Choose experimental files.")
    return fname


def get_base_fname():
    """
    Function that call the window for .nc file selection
    """
    fname = askopenfilename(
        initialdir=os.getcwd(),
        filetypes=(("netcdf file", "*.nc"), ("All Files", "*.*")),
        title="Choose an base file.")
    return fname


expaths = get_exp_fname()               # we read experimental file paths
basepath = get_base_fname()             # we read the base file path
df_brom_base = xr.open_dataset(basepath)


def get_tot_flux(df, var):
    """
    Function get_tot_flux takes xarray dataset and variable name and returns variable total flux array
    in z vertical coordinates. It interpolates fick and sink variable values from z2 to z vertical coordinates
    and add them.
    :param df: input xarray dataset
    :param var: variable name
    :return: variable total flux array in z vertical coordinates
    """
    # z = df['z'].values
    var = re.split(r'_', var)[0]

    fick_z = df['fick:'+var]
    sink_z = df['sink:'+var]

    if var == 'DIC':
        zplot = 1
    else:
        zplot = 45  # Oslofjord
        zplot = 22  # NA

    tot_flux = (fick_z.isel(z2=zplot) + sink_z.isel(z2=zplot))  # NA

    return tot_flux


def get_pCO2(df):
    """
    Calculates pCO2 from .nc file
    :param df: xarray Dataset from nc-file
    :return: pCO2, shape (time, z, i)
    """

    DIC = df['DIC'][:].values
    Kc0 = df['Kc0'][:].values
    Kc1 = df['Kc1'][:].values
    Kc2 = df['Kc2'][:].values
    Hplus = df['Hplus'][:].values

    co2 = DIC / (1 + Kc1 / Hplus + Kc1 * Kc2 / (Hplus ** 2))
    pCO2 = xr.DataArray(co2 / Kc0,
                        {'time': df['time'], 'z': df['z'], 'i': df['i']},
                        name='pCO2')
    return pCO2


def get_pCO2_flux(df, pCO2_atm=400, zlev=2):
    """
    Calculates flux of pCO2 as difference between
    pCO2 in atmosphere and pCO2 at 3rd model level
    :param df: xarray Dataset from nc-file
    :param pCO2_atm: constant pCO2 in atmosphere from fabm.yaml

    :return: pCO2_flux, shape (time, i)
    """
    pCO2 = get_pCO2(df)

    temp = df['T'].values[:, zlev, :]
    windspeed = 8  # from brom.yaml

    # calculate the scmidt number and unit conversions
    Sc = 2073.1 - 125.62 * temp + 3.6276 * temp**2 - 0.043219 * temp**3.0  # 2d array (time, i)
    fwind = (0.222 * windspeed**2 + 0.333 * windspeed) * (Sc/660)**(-0.5)  # 2d array (time, i)
    fwind = fwind * 24/100  # convert to m/day
    pCO2_flux = fwind * (pCO2_atm - (pCO2[:, zlev, :]))  # should be in ppm/(m2d)
    print('min, mean, max')
    print(np.min(fwind), np.mean(fwind), np.max(fwind))
    # flux depends on the difference in partial pressures,
    # wind and henry  here it is rescaled to mmol/m2/d
    dout = xr.DataArray(pCO2_flux,
                        {'time': df['time'], 'i': df['i']},
                         name='flux:pCO2')

    return dout


# noinspection SpellCheckingInspection
def flux_integrate(df_flux, var, dim, timemin=first, timemax=last):
    """
    Function, that calculate and integrate total fluxes over dimension you need.
    Returns integrated dataset (it may be only number)
    df_flux - input xarray with fluxes
    flux - fluxes to calculate and integrate
    dims - dimension to integrate over (time or i)  # FIS: z2 is not implemented???
    """

    if dim == 'i':  # integrate in space (columns, x-values)
        x = df_flux['i'].values
        hx0 = x[1:] - x[:-1]
        hx0 = np.append(hx0, hx0[0])
        cx = np.empty_like(hx0)
        cx[0] = 0
        for i in range(1, len(cx)):
            cx[i] = cx[i-1] + hx0[i-1]/2 + hx0[i]/2
        dx = cx[1:] - cx[:-1]
        dx = np.append(dx, dx[-1] + cx[-1]-cx[-2])
        hx = np.empty_like(hx0)
        hx[0] = dx[-1]/2 + dx[0]/2
        for i in range(1, len(hx)):
            hx[i] = dx[i-1]/2 + dx[i]/2

        if 'time' in df_flux.coords:
            # integrate over columns for a specific time period
            int_flux = np.sum(np.multiply(df_flux.sel(time=slice(timemin, timemax)), hx), axis=-1)
        else:
            # integrate over columns
            int_flux = np.sum(np.multiply(df_flux, hx), axis=-1)

    elif dim == 'time':

        dt = (df_flux['time'][1]-df_flux['time'][0])/np.timedelta64(1, 'D')  # timestep in days fractions
        # integrate over time -> sum over all timesteps (from min to max) * dt
        int_flux = np.sum(df_flux.sel(time=slice(timemin, timemax))*dt.values, axis=0)

    int_flux.name = 'flux:'+re.split(r'_', var)[0]

    return int_flux


def plot_flux(var, flag=flag, dims=dims):
    '''
    Function to draw integrated fluxes
    '''

    if var == 'pCO2':
        df = get_pCO2_flux(df_brom)
        unit = 'ppm'
        print('pCO2 flux was calculated')
    elif var == 'pCO2_diff':
        df = get_pCO2_flux(df_brom)-get_pCO2_flux(df_brom_base)
        unit = 'ppm'
        print('pCO2 flux [experiment - base] was calculated')
    elif re.fullmatch(r'.+_diff', var):
        df = get_tot_flux(df_brom, var)-get_tot_flux(df_brom_base, var)
        unit = '$\mu$M'
        print('total flux [experiment - base] of %s (fick + sink) was calculated' % var)
    else:
        df = get_tot_flux(df_brom, var)
        unit = '$\mu$M'
        print('total flux of %s (fick + sink) was calculated' % var)

    # noinspection SpellCheckingInspection
    if len(dims) == 1:  # check if we integrate only over one dimension - ['i'] or ['time']
        print('Integration only over: current dimensions: ' + str(dims))

        fig, ax = plt.subplots()

        if flag == 'two' and not re.fullmatch(r'.+_diff', var):
            # if we plot experiment and base on one graph, and it is NOT a difference between them!
            # WE PLOT 2 LINES ON ONE PLOT - it's the main idea
            if var == 'pCO2':
                df_base = get_pCO2_flux(df_brom_base)
                # plot baseline
                integral = flux_integrate(df_base, var=var, dim=dims[0])
                integral.plot(ax=ax, c='darkgrey', lw=1.5, label='base')
                # plot experiment
                integral = flux_integrate(df, var=var, dim=dims[0])
                integral.plot(ax=ax, c='k', lw=1.5, alpha=0.8, label='exp')
            else:
                df_base = get_tot_flux(df_brom_base, var)
                # plot baseline
                integral = flux_integrate(df_base, var=var, dim=dims[0])
                integral.plot(ax=ax, c='darkgrey', lw=1.5, label='base')
                # plot experiment
                integral = flux_integrate(df, var=var, dim=dims[0])
                integral.plot(ax=ax, c='k', lw=1.5, alpha=0.8, label='exp')

        else:
            # if it IS difference between experiment and base, or it is just ONE PLOT - EXPERIMENT
            # WE PLOT JUST 1 LINE
            integral = flux_integrate(df, var=var, dim=dims[0])
            integral.plot(ax=ax, c='k', lw=1.5)

        if 'time' in integral.coords:
            # if the variable was integrated over columns or over levels, units = [unit/m]
            ax.set_title(var+', '+unit+'/(m2*day)')
        else:
            # if the variable was integrated over time, units = [unit/m]
            ax.set_title(var+', '+unit+'/day')

        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('')
        ax.tick_params(labelsize=12)
        formatter = mdates.DateFormatter('%d.%m')  # SET DATE FORMAT HERE
        ax.xaxis.set_major_formatter(formatter)
        plt.savefig(re.split(r'_', var)[0]+'\graph_' + var+'_'+re.split('\.', expfile)[0]+'.png',
                    dpi=400, bbox_inches='tight')

def write_numbers(var):
    # Write integral values
    # if we integrate over 2 dimensions
    expfile = re.split('/', expath)[-1]
    if var == 'pCO2':
        df = get_pCO2_flux(df_brom)
        unit = 'ppm'
        print('pCO2 flux was calculated')
    elif var == 'pCO2_diff':
        df = get_pCO2_flux(df_brom)-get_pCO2_flux(df_brom_base)
        unit = 'ppm'
        print('pCO2 flux [experiment - base] was calculated')
    elif re.fullmatch(r'.+_diff', var):
        df = get_tot_flux(df_brom, var)-get_tot_flux(df_brom_base, var)
        unit = '$\mu$M'
        print('total flux [experiment - base] of %s (fick + sink) was calculated' % var)
    else:
        df = get_tot_flux(df_brom, var)
        unit = '$\mu$M'
        print('total flux of %s (fick + sink) was calculated' % var)

    integral = flux_integrate(flux_integrate(df, var=var, dim=dims[0]), var=var, dim=dims[1])
    # print('Number is '+str(round(integral.values, 2)))
    name = var
    # if not os.path.exists(name+'/'):
    #     os.mkdir(name+'/')

    # file = open(name+'/Numbers_'+name+'.txt', 'a')
    file = open('Numbers_' + name + '.txt', 'w')
    file.write('Integral from %s to %s \n' % (first, last))
    file.write(re.split('\.', expfile)[0]+'_'+var+': %.2f' % integral.values + '\n')
    file.close()




for expath in expaths:
    print('file: '+re.split('/', expath)[-1])  # Just for verbose script run
    df_brom = xr.open_dataset(expath)          # convert the data to xarray dataset
    for var in flux_vars:
        print(var)
        plot_flux(var)
        write_numbers(var)
