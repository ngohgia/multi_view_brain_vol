import numpy as np
import matplotlib.pyplot as plt
from nilearn import surface
from nilearn.plotting import plot_surf_stat_map
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colorbar import make_axes

def visualize_volumetric_data_on__multi_view_surf(vol_img, fsaverage=None, threshold=None, vmin=None, vmax=None, title=None, cmap='cold_hot_r'):
    """ Visualizing volumetric data on fsaverage surfaces for the 
        lateral, medial, dorsal and ventral view of both hemispheres
        Parameters
        ----------
        vol_img: brain image in the volumetric space, such as from
            the output of nilearn.image.load_img
        fsaverage: fsaverage surface mesh, either fsaverage5 mesh (10242 nodes),
            or fsaverage mesh (163842 nodes), output from
            nilearn.datasets.fetch_surf_fsaverage()
        threshold : a number or None, default is None.
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image, values
            below the threshold (in absolute value) are plotted as transparent.
        vmin, vmax: lower / upper bound to plot surf_data values
            If None , the values will be set to min/max of the data
        title: str, optional
            Figure title.
        cmap: str, name of the colormap used for visualizing, default is 'cold_hot_r'
    """
    
    lh_mesh = surface.load_surf_mesh(fsaverage.pial_left)
    rh_mesh = surface.load_surf_mesh(fsaverage.pial_right)
        
    lh_surf = surface.vol_to_surf(vol_img, fsaverage.pial_left)
    rh_surf = surface.vol_to_surf(vol_img, fsaverage.pial_right)

    
    if vmin is None:
        vmin = np.min((np.min(lh_surf), np.min(rh_surf)))
    if vmax is None:
        vmax = np.max((np.max(lh_surf), np.max(rh_surf)))

    mesh = [np.vstack((lh_mesh[0], rh_mesh[0])),
            np.vstack((lh_mesh[1], rh_mesh[1] + lh_mesh[1].max(axis=0) + 1))]

    modes = ['lateral', 'medial', 'dorsal', 'ventral']
    hemis = ['left', 'right']
    surf = {'left': lh_mesh, 'right': rh_mesh, 'both': mesh}
    data = {'left': lh_surf, 'right': rh_surf,
               'both': np.hstack((lh_surf, rh_surf))}
    lh_sulc = surface.load_surf_data(fsaverage.sulc_left)
    rh_sulc = surface.load_surf_data(fsaverage.sulc_right)
    bg_map = {'left': lh_sulc, 'right': rh_sulc,
              'both': np.hstack((lh_sulc, rh_sulc))}

    abs_max = np.max((np.abs(vmin), np.abs(vmax)))

    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=(20, 10),  # not the best fix
                             subplot_kw={'projection': '3d'}, dpi=80)

    for index_view, view in enumerate(['lateral', 'medial']):
        for index_hemi, hemi in enumerate(hemis):
            plot_surf_stat_map(surf[hemi], data[hemi],
                               view=view,
                               hemi=hemi,
                               bg_map=bg_map[hemi],
                               axes=axes[index_view, index_hemi],
                               colorbar=False,
                               threshold=threshold,
                               symmetric_cbar=False,
                               vmax=abs_max,
                               cmap=cmap,
                               darkness=0)
            axes[index_view, index_hemi].margins(0)
    for i_m, view in enumerate(['dorsal', 'ventral']):
        plot_surf_stat_map(surf['both'],
                           data['both'],
                           view=view,
                           hemi='left',
                           bg_map=bg_map['both'],
                           axes=axes[i_m, 2],
                           colorbar=False,
                           threshold=threshold,
                           symmetric_cbar=False,
                           vmax=abs_max,
                           cmap=cmap,
                           darkness=0)
        axes[i_m, 2].margins(0)

    true_cmap = get_cmap(cmap)
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    nb_ticks = 5
    ticks = np.linspace(-abs_max, abs_max, nb_ticks)
    bounds = np.linspace(-abs_max, abs_max, true_cmap.N)

    if threshold is not None:
        cmaplist = [true_cmap(i) for i in reversed(range(true_cmap.N))]
        # set colors to grey for absolute values < threshold
        istart = int(norm(-threshold, clip=True) * (true_cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (true_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)
        true_cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, true_cmap.N)

    proxy_mappable = ScalarMappable(cmap=true_cmap, norm=norm)
    proxy_mappable.set_array(np.array([-abs_max, abs_max]))

    cbar_ax, kw = make_axes(axes, location='bottom', fraction=.05,
                         shrink=.3, pad=0.5)
    cbar = fig.colorbar(
        proxy_mappable, cax=cbar_ax, ticks=ticks,
        boundaries=bounds, spacing='proportional',
        format='%.2g', orientation='horizontal')
    cbar.ax.tick_params(labelsize=16) 
    if title is not None:
        fig.suptitle(title, y=0.9, fontsize=20)
    fig.subplots_adjust(wspace=-0.28, hspace=-0.1)

