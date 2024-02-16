import matplotlib.pyplot as plt

N_PLOTS = 3
TEXTWIDTH_LATEX = 219.08614 #pt
TEXTFULLWIDTH_LATEX = 455.244 #pt
# Convert from pt to inches
inches_per_pt = 1 / 72.27

# Golden ratio to set aesthetic figure height
# https://disq.us/p/2940ij3
golden_ratio = (5**.5 - 1) / 2.5

# Figure width in inches
fig_width_in = TEXTWIDTH_LATEX * inches_per_pt
fullfigwidth_in = TEXTFULLWIDTH_LATEX * inches_per_pt
# Figure height in inches
fig_height_in = fig_width_in * golden_ratio * (N_PLOTS)

fig_dim = (fig_width_in * (2/3), fig_height_in)
fig_dim_full = (fullfigwidth_in * (1/3), fig_height_in)
fig_dim_1 = (fig_width_in, fig_height_in / 2)

def plot_coverages_widths(alpha, coverage, width_mean, method, ssc=None, plot_ssc=False, hlines=[], dataset="ds", model="model"):
    """
    Verify coverage, size-stratified class-conditional coverage and width mean)
    """
    if plot_ssc:
        n_plots = 3
    else:
        n_plots = 2
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots*4, 3))
    
    plot_i = 0
    axs[plot_i].scatter(1 - alpha, width_mean, label=method)
    axs[plot_i].set_xlabel("1 - alpha")
    axs[plot_i].set_ylabel("Average size")
    axs[plot_i].set_title('Prediction set size')
    
    plot_i += 1
    if plot_ssc:
        axs[plot_i].scatter(1 - alpha, ssc, label=method)
        axs[plot_i].set_xlabel("1 - alpha")
        axs[plot_i].set_ylabel("Size-stratified Coverage")
        axs[plot_i].plot(1-alpha, 1-alpha, color="black")
        axs[plot_i].set_title('Size-stratified coverage')
        plot_i += 1
    
    axs[plot_i].scatter(1 - alpha, coverage, label=method)
    axs[plot_i].set_xlabel("1 - alpha")
    axs[plot_i].set_ylabel("Coverage")
    axs[plot_i].plot(1-alpha, 1-alpha, color="black")
    axs[plot_i].set_title('Empirical coverage')

    axs[-1].legend()
    plt.tight_layout()
    plt.savefig('results/{}_{}_{}.png'.format(dataset, model, method), dpi=1200)
    plt.show()

def plot_coverages_widths_multi(
        alpha,
        coverages,
        width_means,
        methods,
        legend,
        sscs=None,
        plot_ssc=False,
        hlines=[],
        markers=None,
        dataset="ds",
        model="model",
        title='mytitle'):
    """
    Verify coverage, size-stratified class-conditional coverage and width mean)
    """
    if plot_ssc:
        n_plots = 3
    else:
        n_plots = 2
    if markers is None:
        markers = ['.'] * len(coverages)
    fig, axs = plt.subplots(n_plots, 1, figsize=fig_dim_full, sharex=True)
    c_alpha = .4
    
    sets = zip(coverages, width_means, sscs, methods, markers)

    for (coverage, width_mean, ssc, method, marker) in sets:
        plot_i = 0
        axs[plot_i].scatter(1 - alpha, width_mean, label=method, alpha=c_alpha, marker=marker)
        # axs[plot_i].set_xlabel("1 - alpha")
        axs[plot_i].set_ylabel("Avg. Set Size")
        # axs[plot_i].set_title('Average set Size')
        if len(model) > 0:
            axs[plot_i].set_title(title)
        else:
            axs[plot_i].set_title('{}'.format(dataset))

        plot_i += 1
        if plot_ssc:
            axs[plot_i].scatter(1 - alpha, ssc, label=method, alpha=c_alpha, marker=marker)
            # axs[plot_i].set_xlabel("1 - alpha")
            axs[plot_i].set_ylabel("SSC")
            # axs[plot_i].set_title('Size-stratified coverage')
            plot_i += 1

        axs[plot_i].scatter(1 - alpha, coverage, label=method, alpha=c_alpha, marker=marker)
        axs[plot_i].set_xlabel("1 - alpha")
        axs[plot_i].set_ylabel("Coverage")
        # axs[plot_i].set_title('Coverage')

    for (coverage, width, ssc, label, style) in hlines:
        plot_i = 0
        axs[plot_i].hlines(
            y=coverage,
            xmin=min(1-alpha),
            xmax=max(1-alpha),
            label=label,
            colors='black',
            linestyles=style,
            )
        plot_i += 1
        if plot_ssc:
            axs[plot_i].hlines(
            y=ssc,
            xmin=min(1-alpha),
            xmax=max(1-alpha),
            label=label,
            colors='black',
            linestyles=style,
            )
            plot_i += 1
        axs[plot_i].hlines(
            y=width,
            xmin=min(1-alpha),
            xmax=max(1-alpha),
            label=label,
            colors='black',
            linestyles=style,
            )
    axs[1].plot(1-alpha, 1-alpha, color="black", alpha=c_alpha)
    if plot_ssc:
        axs[2].plot(1-alpha, 1-alpha, color="black", alpha=c_alpha)
    axs[legend[0]].legend(
        ncols=2,
        fontsize='small',
        labelspacing=0.2,
        handletextpad=.3,
        handlelength=1.6,
        borderpad=0.1,
        columnspacing=0.5,
        loc=legend[1],
    )
    plt.tight_layout()
    # plt.suptitle('{} {}'.format(dataset, model))
    plt.savefig('results/{}_{}_multi.png'.format(dataset, model), dpi=1200)
    plt.show()

def plot_cov_x_width_multi(alpha, coverages, width_means, methods, legend, sscs=None, plot_ssc=False, hlines=[], markers=None, dataset="ds", model="model"):
    """
    Verify coverage, size-stratified class-conditional coverage and width mean)
    """
    if plot_ssc:
        n_plots = 2
    else:
        n_plots = 1
    if markers is None:
        markers = ['.'] * len(coverages)
    fig, axs = plt.subplots(n_plots, 1, figsize=fig_dim_1, sharex=True)
    c_alpha = .4
    
    sets = zip(coverages, width_means, sscs, methods, markers)

    for (coverage, width_mean, ssc, method, marker) in sets:
        plot_i = 0
        ax = axs if not plot_ssc else axs[0]
        ax.scatter(coverage, width_mean, label=method, alpha=c_alpha, marker=marker)
        # axs[plot_i].set_xlabel("1 - alpha")
        ax.set_ylabel("Avg. PS size")
        ax.set_xlabel("Coverage")
        # axs[plot_i].set_title('Average set Size')
        if len(model) > 0:
            ax.set_title('{}, {}'.format(dataset, model))
        else:
            ax.set_title('{}'.format(dataset))
        if plot_ssc:
            axs[1].scatter(ssc, width_mean, label=method, alpha=c_alpha, marker=marker)
            # axs[plot_i].set_xlabel("1 - alpha")
            axs[1].set_ylabel("SSC")
            # axs[plot_i].set_title('Size-stratified coverage')
            plot_i += 1
    ax = axs[legend[0]] if plot_ssc else axs
    ax.legend(
        ncols=2,
        fontsize='small',
        labelspacing=0.2,
        handletextpad=.3,
        handlelength=1.6,
        borderpad=0.1,
        columnspacing=0.5,
        loc=legend[1],
    )
    plt.tight_layout()
    # plt.suptitle('{} {}'.format(dataset, model))
    plt.savefig('results/{}_{}_covwidth.png'.format(dataset, model), dpi=1200)
    plt.show()