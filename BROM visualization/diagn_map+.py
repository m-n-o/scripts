# Script for map (x,t) vizualization

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
import os
import re
from tkinter.filedialog import askopenfilename, askopenfilenames
from matplotlib import ticker
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

h = 0.2
w = 0.04
zlev = [3]  # Horizons to draw; o/a flux -> 2; w/b flux -> 45
pCO2_atm = 400

# ATTENTION! Fluxes must be named only 'Variable_tot_flux' or 'pCO2_flux' and takes only one name
varname = ['MgOH2_tot_flux']


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


def get_pCO2_flux(df, pCO2_atm):
    """
    Calculates flux of pCO2 as difference between
    pCO2 in atmosphere and pCO2 at 3rd model level
    :param df: xarray Dataset from nc-file
    :param pCO2_atm: constant pCO2 in atmosphere from fabm.yaml
    :return: pCO2_flux, shape (time, i)
    """
    pCO2 = get_pCO2(df)
    pCO2_flux = pCO2_atm - pCO2[:, 2, :]

    return pCO2_flux


def get_tot_flux(df, var):
    """
    Function get_TotFlux takes xarray dataset and variable name and returns variable total flux array
    in z vertical coordinates. It interpolates fick and sink variable values from z2 to z vertical coordinates and add them.
    :param df: input xarray dataset
    :param var: variable name
    :return: variable total flux array in z vertical coordinates
    """
    # z = df['z'].values
    only_var = re.split(r'_', var)[0]  # Varname without tail after '_'

    fick_z = df['fick:'+only_var]
    sink_z = df['sink:'+only_var]
    tot_flux = (fick_z + sink_z)

    return tot_flux


def make_fig_gs(nrows, ncols):
    global fig
    fig = plt.figure(figsize=(5*ncols, 3*nrows), dpi=100)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(left=0.07, right=0.93,
              bottom=0.08, top=0.95,
              wspace=0.4, hspace=0.7)
    return gs


def sbplt(to_gs):
    return [fig.add_subplot(a) for a in to_gs]


expaths = get_exp_fname()               # we read experimental file paths
basepath = get_base_fname()             # we read the base file path


def plot_param(var, x, y, axis, zlev, df_brom, df_brom_base):
    if var == 'pCO2':
        var = get_pCO2(df_brom)[:, zlev, :]
    elif var == 'pCO2_flux':
        var = get_pCO2_flux(df_brom, pCO2_atm=pCO2_atm)
    elif var == 'pCO2_flux_diff':
        var = (get_pCO2_flux(df_brom, pCO2_atm=pCO2_atm) - get_pCO2_flux(df_brom_base, pCO2_atm=pCO2_atm))
    elif re.fullmatch(r'.+tot_flux', var):
        var = get_tot_flux(df_brom, var)[:, zlev, :]
    elif re.fullmatch(r'.+flux_diff', var):
        var = (get_tot_flux(df_brom, var)[:, zlev, :] - get_tot_flux(df_brom_base, var)[:, zlev, :])
    else:
        var = df_brom[var][:, zlev, :]   # retrieve variable from xarray dataset
    levels = np.linspace(np.min(var), np.max(var), 30)
    if all(lev == levels[0] for lev in levels):
        levels = np.linspace(levels[0], levels[0] + 0.1, 30)

    X, Y = np.meshgrid(x, y)
    cmap = plt.get_cmap('jet')

    CS_1 = axis.contourf(X, Y, var.squeeze().T, levels=levels, extend="both", cmap=cmap)

    tick_locator = ticker.MaxNLocator(nbins=4)
    cb = plt.colorbar(CS_1, ax=axis)
    cb.locator = tick_locator
    cb.update_ticks()

    datemin = np.datetime64('2012-01-03')  # Start time 'YYYY-MM-DD' '2012-01-03'
    datemax = np.datetime64('2012-01-31')  # Finish time 'YYYY-MM-DD'
    axis.set_xlim(datemin, datemax)
    axis.tick_params(labelsize=12)
    axis.set_xlabel('Days', fontsize=12)
    axis.set_ylabel('Distance, m', fontsize=12)

    # format the coords message box
    formatter = mdates.DateFormatter('%d.%m')
    axis.xaxis.set_major_formatter(formatter)


# Map
def fig_map(expath, varname, zlev,
            nrows=1, ncols=1):

    df_brom = xr.open_dataset(expath)         # convert the data to xarray dataset
    df_brom_base = xr.open_dataset(basepath)
    xs = df_brom['time'].values                # read time for horizontal axis
    ys = df_brom['i'].values                   # read i position (distance) for vertical axis

    gs = make_fig_gs(nrows, ncols)
    ax = sbplt(list(gs))

    for i in np.arange(len(varname)):
        plot_param(varname[i], xs, ys, ax[i], zlev, df_brom, df_brom_base)

        if varname[i] in ['pH', 'Om_Ar', 'Om_MgOH2']:
            print(i, varname[i])
            ax[i].set_title('%s' % varname[i])
        elif varname[i] == 'pCO2':
            print(i, varname[i])
            ax[i].set_title('%s, ppm' % varname[i])
        elif varname[i] in ['pCO2_flux', 'pCO2_flux_diff']:
            print(i, varname[i])
            ax[i].set_title('%s, ppm/m$^{2}$*day' % varname[i])
        elif varname == 'MgOH2_mg_l':
            print(i, varname[i])
            ax[i].set_title('%s, mg/l' % varname[i])
        elif re.fullmatch(r'.+tot_flux', varname[i]):
            ax[i].set_title('%s, $\mu$M/m$^{2}$*day' % varname[i])
        elif re.fullmatch(r'.+flux_diff', varname[i]):
            ax[i].set_title('%s, $\mu$M/m$^{2}$*day' % varname[i])
        else:
            print(i, varname[i])
            ax[i].set_title('%s, $\mu M$' % varname[i])

    name = re.split(r'_', varname[0])[0]
    if not os.path.exists(name+'/'):
        os.mkdir(name+'/')

    expfile = re.split('/', expath)[-1]
    plt.savefig(name+'\map_'+varname[0]+'_'+re.split('\.', expfile)[0]+'.png',
                bbox_inches='tight')


if __name__ == '__main__':
    for expath in expaths:
        print('file: '+re.split('/', expath)[-1])  # Just for verbose script run
        fig_map(expath, varname, zlev)  # Path to experimental file, varname and vertical horizon
