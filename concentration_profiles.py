import xarray as xr
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import gsw

plt.rcParams["figure.facecolor"] = 'w'

iday = 365*7+100  # drawn day
hor_ax = 'dens'  # Horizontal axis. Only z or dens.

# drawn concentrations
vax = [['T', 'S', 'O2'],
       ['H2S', 'SO4', 'S0'],
       ['NO2', 'NO3', 'NH4'],
       ['Mn4', 'Mn3', 'Mn2'],
       ['Fe3', 'Fe2'],
       ['Phy', 'Het'],
       ['Baae', 'Bhae', 'Baan', 'Bhan'],
       ['DOML', 'DOMR', 'POML', 'POMR'],
       ['PO4', 'Si', 'CH4'],
       ['pH', 'Alk', 'DIC']
       ]

# drawn concentration colors
colors_vax = [['#377eb8', '#e41a1c', '#4daf4a'],
              ['#984ea3', '#a65628', '#e6ab02'],
              ['#377eb8', '#e41a1c', '#f781bf'],
              ['#377eb8', '#4daf4a', '#984ea3'],
              ['#a65628', '#f781bf'],
              ['#4daf4a', '#ff7f00'],
              ['#377eb8', '#4daf4a', '#984ea3', '#e6ab02'],
              ['#a65628', '#f781bf', '#e41a1c', '#377eb8'],
              ['#4daf4a', '#984ea3', '#ff7f00'],
              ['#e6ab02', '#a65628', '#f781bf']
              ]


def get_fname():
    fname = askopenfilename(
        initialdir=os.getcwd(),
        filetypes=(("netcdf file", "*.nc"), ("All Files", "*.*")),
        title="Choose a needed file.")
    return fname


fname = get_fname()
ds = xr.open_dataset(fname)

if hor_ax == 'z':
    depth = ds['z']  # depth
elif hor_ax == 'dens':
        P = ds['z'] - 10.1325  # Sea pressure in dbar
        SA = gsw.SA_from_SP(ds['S'].isel(time=iday),  # Absolute salinity
                    P, 42, 30.5)
        CT = gsw.CT_from_t(SA, ds['T'].isel(time=iday), P)  # Conservative temperature (ITS-90)
        depth = gsw.density.sigma0(SA, CT)  # potential density anomaly
else:
    print('Variable hor_ax is incorrect. Check it, please.')

fig, axs = plt.subplots(2, 5, figsize=(15, 15), gridspec_kw={'hspace': 0.6, 'wspace': 0.15})
for i, ax in enumerate(axs.ravel()):
    ax_vars, ax_colors = vax[i], colors_vax[i]
    ax.invert_yaxis()
    shift = 0
    ax.grid(which='major', axis='y', linestyle='--', color='0.5')
    if i not in [0, 5]:
        ax.tick_params(left=False, labelleft=False)
    else:
        if hor_ax == 'z':
            ax.set_ylabel('Depth, m', fontsize=13)
        else:
            ax.set_ylabel('Sigma', fontsize=13)
        ax.tick_params(axis='y', labelsize=13)

    for av, ac in zip(ax_vars, ax_colors):
        print(av)
        var = ds[av][37, :, 0]
        axn = ax.twiny()
        axn.plot(var, depth, zorder=5, color=ac, lw=1.5)
        axn.spines['top'].set_position(('outward', shift))
        shift += 41
        axn.tick_params(axis='x', labelcolor=ac, labelsize=13)
        ax.tick_params(bottom=False, labelbottom=False)
        if av == 'T':
            axn.set_xlabel(r'T, $^{\circ}$C', color=ac, fontsize=13)
        elif av == 'S':
            axn.set_xlabel('S, PSU', color=ac, fontsize=13)
        elif av == 'pH':
            axn.set_xlabel('pH', color=ac, fontsize=13)
        elif av == 'pCO2':
            axn.set_xlabel('pCO2, ppm', color=ac, fontsize=13)
        else:
            axn.set_xlabel(av+r', $\mu$M', color=ac, fontsize=13)

# plt.show()
plt.savefig('Conc_profiles.png', dpi=400, bbox_inches='tight')
