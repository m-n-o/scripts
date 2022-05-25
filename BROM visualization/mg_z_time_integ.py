import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import re
from tkinter.filedialog import askopenfilename, askopenfilenames
from matplotlib import ticker
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

h = 0.3
w = 0.1
sed = 45  # ???
sed2 = 44

first = np.datetime64('2012-01-03')  # there should be the 1st day you want to draw
last = np.datetime64('2012-01-31')   #('2019-12-31') #there should be the last day you want to draw

varname = ['MgOH2_diff']  # takes only one name


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


def make_fig_gs(nrows, ncols):
    global fig
    fig = plt.figure(figsize=(5*ncols, 2*nrows), dpi=100)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(left=0.07, right=0.93,
              bottom=0.08, top=0.95,
              wspace=0.4, hspace=0.5)
    return gs


def create_gs(pos):
    return gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[pos],
                                            hspace=h, wspace=w, height_ratios=[3, 2],
                                            width_ratios=[15, 1])


def call_create_gs(poss):
    return [create_gs(n) for n in poss]


def sbplt_cb(to_gs):
    return [fig.add_subplot(a) for a in to_gs]


expaths = get_exp_fname()               # we read experimental file paths
basepath = get_base_fname()             # we read the base file path


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


def integr_x(dataarray, var, timemin=first, timemax=last):
    """
    Function to integrate over x coordinate
    :param dataarray: DataArray (time, i) with values
    :param var: variable name
    :param timemin: first time level
    :param timemax: last time level
    :return: inted, DataArray (time)
    """

    x = dataarray['i'].values
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

    inted = np.sum(np.multiply(dataarray.sel(time=slice(timemin, timemax)), hx), axis=-1)/hx.sum()
    inted.name = var

    return inted


def plot_param(df1, df2, name, axis, axis_cb, axis_sed, axis_cb_sed,
               datemin=first, datemax=last):

    if name == 'pCO2':
        var = integr_x(get_pCO2(df1), name)
    elif name == 'diff_pCO2':
        var = (integr_x(get_pCO2(df1), name) - integr_x(get_pCO2(df2), name))
    elif re.fullmatch(r'.+_diff', name):
        only_var = re.split(r'_', name)[0]
        var = (integr_x(df1[only_var], only_var) - integr_x(df2[only_var], only_var))
    else:
        var = integr_x(df1[name], name)

    levels = np.linspace(np.min(var[:, :sed]), np.max(var[:, :sed]), 30)
    sed_levels = np.linspace(np.min(var[:, sed2:]), np.max(var[:, sed2:]), 30)
    if all(lev == levels[0] for lev in levels):
        levels = np.linspace(levels[0], levels[0]+0.1, 30)
    if all(lev == sed_levels[0] for lev in sed_levels):
        sed_levels = np.linspace(sed_levels[0], sed_levels[0]+0.1, 30)

    X, Y = np.meshgrid(var['time'], var['z'][:sed])
    X_sed, Y_sed = np.meshgrid(var['time'], var['z'][sed2:])
    cmap = plt.get_cmap('jet')

    CS_1 = axis.contourf(X, Y, var[:, :sed].T, levels=levels, extend="both", cmap=cmap)
    if var[:, sed2:].shape[1] == len(Y_sed):
        CS_1_sed = axis_sed.contourf(X_sed, Y_sed, var[:, sed2:].T, levels=sed_levels, extend="both", cmap=cmap)
    else:
        CS_1_sed = axis_sed.contourf(X_sed, Y_sed, var[:, sed2+1:].T, levels=sed_levels, extend="both", cmap=cmap)

    tick_locator = ticker.MaxNLocator(nbins=4)

    cb = plt.colorbar(CS_1, cax=axis_cb)

    cb.locator = tick_locator
    cb.update_ticks()

    cb_sed = plt.colorbar(CS_1_sed, cax=axis_cb_sed)

    cb_sed.locator = tick_locator
    cb_sed.update_ticks()

    axis.set_ylim(np.max(var['z'][:sed]), 0)
    axis_sed.set_ylim(np.max(var['z'][sed2:]), np.min(var['z'][sed2:]))

    axis_sed.axhline(0, linestyle='--', linewidth=0.5, color='w')

    axis.tick_params(axis='y', pad=0.01)
    axis_sed.tick_params(axis='y', pad=1)

    # years = mdates.DayLocator(interval=2)  # every 2nd day
    # years_fmt = mdates.DateFormatter('%D')
    #
    # axis.xaxis.set_major_locator(years)
    # axis.xaxis.set_major_formatter(years_fmt)
    # axis_sed.xaxis.set_major_locator(years)
    # axis_sed.xaxis.set_major_formatter(years_fmt)

    axis.set_xlim(datemin, datemax)
    axis_sed.set_xlim(datemin, datemax)
    # format the coords message box
    formatter = mdates.DateFormatter('%d.%m')
    axis_sed.xaxis.set_major_formatter(formatter)
    # axis_sed.tick_params(axis='x', labelrotation=45)
    axis.set_xticklabels([])  # check it
    axis.tick_params(labelsize=12)
    axis_sed.tick_params(labelsize=12)
    axis_sed.set_xlabel('Days', fontsize=12)
    axis.set_ylabel('Depth, m', fontsize=12)


def fig_ztime(expath, varname, nrows=1, ncols=1):
    global gs

    df_brom = xr.open_dataset(expath)         # convert the data to xarray dataset
    df_brom_base = xr.open_dataset(basepath)

    gs = make_fig_gs(nrows, ncols)
    gss = call_create_gs(np.arange(len(varname)))

    axes = []
    axes_cb = []
    axes_sed = []
    axes_sed_cb = []
    for g in gss:
        ax, ax_cb, ax_sed, ax_sed_cb = sbplt_cb(list(g))
        axes.append(ax)
        axes_cb.append(ax_cb)
        axes_sed.append(ax_sed)
        axes_sed_cb.append(ax_sed_cb)

    for i in np.arange(len(varname)):
        print(i, varname[i])
        if varname[i] != 'Kz':
            plot_param(df_brom, df_brom_base, varname[i], axes[i], axes_cb[i], axes_sed[i], axes_sed_cb[i])
        else:
            plot_param(df_brom, df_brom_base, varname[i], axes[i], axes_cb[i], axes_sed[i], axes_sed_cb[i])
        axes[i].set_title('%s, $\mu M$' % varname[i])

    name = re.split(r'_', varname[0])[0]
    if not os.path.exists(name+'/'):
        os.mkdir(name+'/')

    expfile = re.split('/', expath)[-1]
    plt.savefig(name+'\sec_' + varname[0]+'_'+re.split('\.', expfile)[0]+'.png',
                bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    for expath in expaths:
        print('file: '+re.split('/', expath)[-1])  # Just for verbose script run
        fig_ztime(expath, varname)  # Path to experimental file and varname
