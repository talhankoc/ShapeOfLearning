import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import inf

def plot_dgms(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,	
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None,
    save_path=None
):
    """A helper function to plot persistence diagrams. 
    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with
        
        .. code:: python
            import matplotlib as mpl
            print(mpl.styles.available)
    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    mpl.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()
    
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.close()


#-----------------------------------------#
path = '/Users/tkoc/Code/ShapeOfLearning/Homology/FloydVRInputLayer/'
file_name = 'VRFiltration_BettiData.txt'
folder_prefix = 'Fashion2_'
layer_sizes = [8,16,24,32,40,48]
epochs = 50


#FloydVRInputLayer
layer_xy_values = [29.518257, 19.735643, 19.655315, 20.1538, 19.606615, 19.76956]

# FloydVROutputLayer
#layer_xy_values = [4.516148, 5.142353, 5.3599305, 5.422422, 6.338068, 6.795538]

#
#layer_xy_values = [32.5, 20.5, 16.5, 15.9, 14.3, 14.5]



for layer_size, x_range in zip(layer_sizes,layer_xy_values):
	for epoch in range(1, epochs+1):
		folder_path = path + folder_prefix + str(epoch) + '_' + str(layer_size) + '/'
		data_path = folder_path + file_name
		with open(data_path, 'rb') as f:
			diagrams = pickle.load(f)

			#calculate and save some basic facts about betti persistance
			h0_total = len(diagrams[0])
			h1_total = len(diagrams[1])
			#h2_total = len(diagrams[2])

			avg_h0_life = sum([ y-x for x,y in diagrams[0] if y != inf ])/h0_total
			avg_h1_life = sum([ y-x for x,y in diagrams[1] if y != inf ])/h1_total
			# if h2_total != 0:
			# 	avg_h2_life = sum([ y-x for x,y in diagrams[2] if y != inf ])/h2_total
			# else:
			# 	avg_h2_life = 0

			with open(folder_path + 'analysis.txt', 'wb') as f2:
				pickle.dump([h0_total,h1_total, avg_h0_life,avg_h1_life], f2)

			
			print('Layer',layer_size, '\tepoch', epoch)
			plot_dgms(diagrams, 
				size=12,
				title='Layer Size='+str(layer_size)+', Epoch='+str(epoch),
				save_path=folder_path+'birth_death.png', 
				xy_range=[-1,x_range,-1,x_range])
			plot_dgms(diagrams, 
				size=12, 
				title='Layer Size='+str(layer_size)+', Epoch='+str(epoch),
				save_path=folder_path+'lifetime.png', 
				xy_range=[-1,x_range,-1,x_range], lifetime=True,)
