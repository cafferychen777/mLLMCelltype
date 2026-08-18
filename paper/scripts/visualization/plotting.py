import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from adjustText import adjust_text
from pylab import cm
from matplotlib import colors

def PCA_var_explained_plots(adata):
        n_rows = 1
        n_cols = 2
        fig = plt.figure(figsize=(n_cols*4.5, n_rows*3))
        # variance explained
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        x1 = range(len(adata.uns['pca']['variance_ratio']))
        y1 = adata.uns['pca']['variance_ratio']
        ax1.scatter(x1, y1, s=3)
        ax1.set_xlabel('PC'); ax1.set_ylabel('Fraction of variance explained')
        ax1.set_title('Fraction of variance explained per PC')
        # cum variance explainend
        ax2 = fig.add_subplot(n_rows, n_cols, 2)
        cml_var_explained = np.cumsum(adata.uns['pca']['variance_ratio'])
        x2 = range(len(adata.uns['pca']['variance_ratio']))
        y2 = cml_var_explained
        ax2.scatter(x2, y2, s=4)
        ax2.set_xlabel('PC')
        ax2.set_ylabel('Cumulative fraction of variance explained')
        ax2.set_title('Cumulative fraction of variance explained by PCs')
        plt.tight_layout()
        plt.show()


def assign_to_red_or_black_group(x, y, x_cutoff, y_cutoff):
    """xcoord is coefficient (MAST already took log2). ycoord is -log10(pval). label is gene name."""
    if abs(x) > x_cutoff and y > y_cutoff:
        color = "red"
    # x coordinate (coef) is set to 0 if one of the two groups has zero counts (in that case,
    # a fold change cannot be calculated). We'll color these points with 'salmon' (similar to red)
    elif abs(x) == 0 and y > y_cutoff:
        color = "salmon"
    else:
        color = "black"
    return color


def plot_volcano_plot(
    dea_results,
    x_cutoff,
    y_cutoff,
    title,
    use_zscores=False,
    plot_labels=True,
    min_red_dots=None,
    figsize=(15, 7.5),
    show_plot=False,
):
    """makes volcano plot. title is title of plot. path is path to MAST output csv. cutoffs will determine
    which dots will be colored red. plot_labels can be set to False if no labels are wanted, otherwise all
    red dots will be labeled with their gene name. If min_red_dots is set to a number, the x_cutoff will be
    decreased (with factor .9 every time) until at least min_red_dots are red. figsize is a tuple of size 2,
    and determines size of the figure. Returns the figure."""
    coefs = dea_results.loc[:, "coef"].copy()
    xcoords = coefs.fillna(0)
    if use_zscores:
        pvals = dea_results.loc[:, "coef_Z"]
        ycoords = pvals
    else:
        pvals = dea_results.loc[:, "pval_adj"].copy()
        # NOTE: SETTING PVALS TAHT ARE 0 (DUE TO ROUNDING) TO MINIMUM NON ZERO VALUE HERE
        pvals[pvals == 0] = np.min(pvals[pvals != 0])  # np.nextafter(0, 1)
        ycoords = -np.log10(pvals)
    gene_names = dea_results.index.tolist()
    colors = [
        assign_to_red_or_black_group(x, y, x_cutoff, y_cutoff)
        for x, y in zip(xcoords, ycoords)
    ]

    # if min_red_dots is set (i.e. not None), check if enough points are labeled red. If not, adjust x cutoff:
    if min_red_dots != None:
        n_red_points = sum([x == "red" for x in colors])
        while n_red_points < min_red_dots:
            x_cutoff = 0.9 * x_cutoff  # make x cutoff less stringent
            # reevaluate color of points using new cutoff:
            colors = [
                assign_to_red_or_black_group(x, y, x_cutoff, y_cutoff)
                for x, y in zip(xcoords, ycoords)
            ]
            n_red_points = sum([x == "red" for x in colors])
    # extract coordinates separately for red and black
    black_coords = [
        (x, y) for x, y, color in zip(xcoords, ycoords, colors) if color == "black"
    ]
    red_coords = [
        (x, y) for x, y, color in zip(xcoords, ycoords, colors) if color == "red"
    ]
    salmon_coords = [
        (x, y) for x, y, color in zip(xcoords, ycoords, colors) if color == "salmon"
    ]

    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(
        [x for x, y in black_coords],
        [y for x, y in black_coords],
        marker=".",
        linestyle="",
        color="royalblue",
    )
    plt.plot(
        [x for x, y in salmon_coords],
        [y for x, y in salmon_coords],
        marker=".",
        linestyle="",
        color="salmon",
    )
    plt.plot(
        [x for x, y in red_coords],
        [y for x, y in red_coords],
        marker=".",
        linestyle="",
        color="red",
    )
    if plot_labels == True:
        ten_lowest_salmon_pvals_gene_names = [
            gene_name
            for _, gene_name, color in sorted(zip(pvals, gene_names, colors))
            if color == "salmon"
        ][:10]
        # label if color is set to red, or if color is set to salmon and the salmon color is one of the ten salmon genes with lowest pval
        labels = [
            plt.text(x, y, label, ha="center", va="center")
            for x, y, color, label in zip(xcoords, ycoords, colors, gene_names)
            if (
                color in ["red"]
                or (color == "salmon" and label in ten_lowest_salmon_pvals_gene_names)
            )
        ]
        adjust_text(labels)
    plt.xlabel(
        "coef (=log(fold chagne))",
        fontsize=13,
    )
    if use_zscores:
        plt.ylabel("Z-score based on stdev")
    else:
        plt.ylabel("-log10 adjusted p-value", fontsize=14)
    plt.title(
        title
        + " (n genes: "
        + str(len(gene_names))
        + ") \n x-cutoff="
        + str(round(x_cutoff, 2))
        + ", y-cutoff="
        + str(round(y_cutoff, 2)),
        fontsize=16,
    )
    if show_plot == False:
        plt.close()
    return fig

def plot_bar_chart(
    adata,
    x_var,
    y_var,
    x_names=None,
    y_names=None,
    y_min=0,
    return_fig=False,
    cmap="tab20",
):
    """plots stacked bar chart.
    Arguments
        adata - anndata object
        x_var - name of obs variable to use for x-axis
        y_var - name of obs variable to use for y-axis
        x_names - names of x groups to include, exclude all other groups
	y_names - names of y groups to include, exclude all other groups
	y_min - minimum percentage of group to be labeled in plots. If
		percentage of a y_group is lower than this minimum in all
		x_groups, then the y_group will be pooled under "other".
        return_fig - (Boolean) whether to return matplotlib figure
        cmap - name of matplotlib colormap

    Returns:
        matplotlib figure of barchart if return_fig is True. Otherwise nothing.
    """
    bar_chart_df_abs = adata.obs.groupby([x_var, y_var]).agg(
        {x_var: "count"}
    )  # calculate count of each y_var for each x_var
    bar_chart_df = (
        bar_chart_df_abs.groupby(level=0)
        .apply(lambda x: x / float(x.sum()) * 100)
        .unstack()
    )  # convert to percentages
    # clean up columns/index
    bar_chart_df.columns = bar_chart_df.columns.droplevel(0)
    bar_chart_df.index.name = None
    bar_chart_df.columns.name = None
    # if y_min > 0, re-map y categories:
    if y_min > 0:
        # check which y variables never have a fraction above y_min
        y_var_to_remove = (bar_chart_df >= y_min).sum(axis=0) == 0
        y_var_remapping = dict()
        for y_name, to_remove in zip(y_var_to_remove.index, y_var_to_remove.values):
            if to_remove:
                y_var_remapping[y_name] = "other"
            else:
                y_var_remapping[y_name] = y_name
        adata.obs["y_temp"] = adata.obs[y_var].map(y_var_remapping)
        # recalculate bar_chart_df, now using re-mapped y_var
        bar_chart_df_abs = adata.obs.groupby([x_var, "y_temp"]).agg(
            {x_var: "count"}
        )  # calculate count of each y_var for each x_var
        bar_chart_df = (
            bar_chart_df_abs.groupby(level=0)
            .apply(lambda x: x / float(x.sum()) * 100)
            .unstack()
        )  # convert to percentages
        # clean up columns/index
        bar_chart_df.columns = bar_chart_df.columns.droplevel(0)
        bar_chart_df.index.name = None
        bar_chart_df.columns.name = None
    # prepare x and y variables for bar chart:
    if x_names is None:
        x_names = bar_chart_df.index
    else:
        if not set(x_names).issubset(adata.obs[x_var]):
            raise ValueError("x_names should be a subset of adata.obs[x_var]!")
    if y_names is None:
        y_names = bar_chart_df.columns
    else:
        if not set(y_names).issubset(adata.obs[y_var]):
            raise ValueError(
                "y_names should be a subset of adata.obs[y_var]! (Note that this can be affected by your y_min setting.)"
            )
    # subset bar_chart_df based on x and y names:
    bar_chart_df = bar_chart_df.loc[x_names, y_names]
    x_len = len(x_names)
    y_names = bar_chart_df.columns
    y_len = len(y_names)
    # setup colors
    colormap = cm.get_cmap(cmap)
    cols = [colors.rgb2hex(colormap(i)) for i in range(colormap.N)]
    # set bar width
    barWidth = 0.85
    # plot figure
    fig = plt.figure(figsize=(12, 3))
    axs = []
    # plot the bottom bars of the stacked bar chart
    axs.append(
        plt.bar(
            range(len(x_names)),
            bar_chart_df.loc[:, y_names[0]],
            color=cols[0],
            # edgecolor="white",
            width=barWidth,
            label=y_names[0],
        )
    )
    # store the bars as bars_added, to know where next stack of bars should start
    # in y-axis
    bars_added = [bar_chart_df.loc[:, y_names[0]]]
    # now loop through the remainder of the y categories and plot
    for i, y in enumerate(y_names[1:]):
        axs.append(
            plt.bar(
                x=range(len(x_names)),  # numbers of bars [1, ..., n_bars]
                height=bar_chart_df.loc[:, y],  # height of current stack
                bottom=[
                    sum(idx_list) for idx_list in zip(*bars_added)
                ],  # where to start current stack
                color=cols[i + 1],
                # edgecolor="white",
                width=barWidth,
                label=y,
            )
        )
        # append plottend bars to bars_added variable
        bars_added.append(bar_chart_df.loc[:, y])
    # Custom x axis
    plt.xticks(range(len(x_names)), x_names, rotation=90)
    plt.xlabel(x_var)
    # Add a legend
    plt.legend(
        axs[::-1],
        [ax.get_label() for ax in axs][::-1],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
    )
    # add y label:
    plt.ylabel("percentage of cells")
    # add title:
    plt.title(f"{y_var} fractions per {x_var} group")
    # Show graphic:
    plt.show()
    # return figure:
    if return_fig:
        return fig



def plot_dataset_statistics(
    adata, return_fig=False, show=True, fontsize=10, figwidthscale=3, figheightscale=4
):
    data_by_subject = adata.obs.groupby("subject_ID").agg(
        {
            "study": "first",
        }
    )
    data_by_sample = adata.obs.groupby("sample").agg({"study": "first"})
    n_figures = 3
    n_cols = 3
    n_rows = int(np.ceil(n_figures / n_cols))
    fig = plt.figure(figsize=(figwidthscale * n_cols, figheightscale * n_rows))
    fig_count = 0
    # FIGURE
    fig_count += 1
    ax = fig.add_subplot(n_rows, n_cols, fig_count)
    dataset_subj_freqs = data_by_subject.study.value_counts()
    datasets_ordered = dataset_subj_freqs.index
    ax.bar(dataset_subj_freqs.index, dataset_subj_freqs.values)
    ax.set_title("subjects per study", fontsize=fontsize)
    ax.set_ylabel("n subjects", fontsize=fontsize)
    ax.tick_params(axis="x", rotation=90, labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(False)
    # FIGURE
    fig_count += 1
    ax = fig.add_subplot(n_rows, n_cols, fig_count)
    dataset_sample_freqs = data_by_sample.study.value_counts()
    ax.bar(datasets_ordered, dataset_sample_freqs[datasets_ordered].values)
    ax.set_title("samples per study", fontsize=fontsize)
    ax.set_ylabel("n samples", fontsize=fontsize)
    ax.tick_params(axis="x", rotation=90, labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(False)
    # FIGURE
    fig_count += 1
    ax = fig.add_subplot(n_rows, n_cols, fig_count)
    dataset_cell_freqs = adata.obs.study.value_counts()
    ax.bar(datasets_ordered, dataset_cell_freqs[datasets_ordered].values)
    ax.set_title("cells per study", fontsize=fontsize)
    ax.set_ylabel("n cells", fontsize=fontsize)
    ax.tick_params(axis="x", rotation=90, labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(False)
    plt.tight_layout()
    plt.grid(False)
    if show:
        plt.show()
    plt.close()
    if return_fig:
        return fig

def plot_subject_statistics(
    adata,
    return_fig=False,
    show=True,
    fontsize=12,
    figheight=5,
    figwidth=5,
    barwidth=0.10,
):
    data_by_subject = adata.obs.groupby("subject_ID").agg(
        {
            "age": "first",
            "BMI": "first",
            "ancestry": "first",
            "sex": "first",
            "smoking_status": "first",
        }
    )
    fig = plt.figure(
        figsize=(figwidth, figheight),
        constrained_layout=True,
    )
    gs = GridSpec(12, 12, figure=fig)
    fig_count = 0
    # FIGURE 1 AGE
    fig_count += 1
    ax = fig.add_subplot(gs[:6, :6])
    bins = np.arange(0, max(adata.obs.age), 5)
    tick_idc = np.arange(0, len(bins), 4)
    perc_annotated = int(
        np.round(
            100 - (data_by_subject.age.isnull().sum() / data_by_subject.shape[0] * 100),
            0,
        )
    )
    ax.hist(data_by_subject.age, bins=bins, rwidth=0.9)
    print(f"age: {perc_annotated}% annotated")
    ax.set_xlabel("age", fontsize=fontsize)
    ax.set_xticks(bins[tick_idc])
    ax.tick_params(labelsize=fontsize, bottom=True, left=True)
    ax.set_ylabel("n subjects", fontsize=fontsize)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # FIGURE 2 BMI
    fig_count += 1
    ax = fig.add_subplot(gs[:6, -6:])
    BMIs = data_by_subject.BMI.copy()
    perc_annotated = int(round(100 - (BMIs.isna().sum() / len(BMIs) * 100)))
    BMIs = BMIs[~BMIs.isna()]
    bins = np.arange(np.floor(BMIs.min() / 2) * 2, BMIs.max(), 2)
    tick_idc = np.arange(0, len(bins), 3)
    ax.hist(data_by_subject.BMI, bins=bins, rwidth=0.9)
    print(f"BMI: {perc_annotated}% annotated")
    ax.set_xlabel("BMI", fontsize=fontsize)
    ax.set_ylabel("n subjects", fontsize=fontsize)
    ax.set_xticks(bins[tick_idc])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize, bottom=True, left=True)
    ax.grid(False)
    # FIGURE 3 SEX
    fig_count += 1
    ax = fig.add_subplot(gs[-6:, :3])
    x_man = np.sum(data_by_subject.sex == "male")
    x_woman = np.sum(data_by_subject.sex == "female")
    perc_annotated = int(
        np.round(
            100
            - sum([s == "nan" or pd.isnull(s) for s in data_by_subject.sex])
            / data_by_subject.shape[1]
            * 100,
            0,
        )
    )
    ax.bar(
        x=[0.25, 0.75],
        tick_label=["male", "female"],
        height=[x_man, x_woman],
        width=barwidth * 5 / 3,
    )
    ax.set_xlim(left=0, right=1)
    print(f"sex: {perc_annotated}% annotated)")
    ax.tick_params("x", rotation=90, labelsize=fontsize, bottom=True, left=True)
    ax.tick_params("y", labelsize=fontsize, bottom=True, left=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("n subjects", fontsize=fontsize)
    ax.set_xlabel("sex", fontsize=fontsize)
    ax.grid(False)
    # FIGURE 4 ANCESTRY
    fig_count += 1
    ax = fig.add_subplot(gs[-6:, 3:-4])
    ethns = data_by_subject.ancestry.copy()
    perc_annotated = int(
        np.round(
            100 - sum([e == "nan" or pd.isnull(e) for e in ethns]) / len(ethns) * 100, 0
        )
    )
    ethns = ethns[ethns != "nan"]
    ethn_freqs = ethns.value_counts()
    n_bars = len(ethn_freqs)
    ax.bar(
        x=np.linspace(0 + 0.75 / n_bars, 1 - 0.75 / n_bars, n_bars),
        tick_label=ethn_freqs.index,
        height=ethn_freqs.values,
        width=barwidth,
    )
    ax.set_xlim(left=0, right=1)
    print(f"Ancestry {perc_annotated}% annotated")
    ax.set_ylabel("n subjects", fontsize=fontsize)
    ax.set_xlabel("Ancestry", fontsize=fontsize)
    ax.tick_params("x", rotation=90, labelsize=fontsize, bottom=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params("y", labelsize=fontsize, left=True)
    ax.grid(False)
    # FIGURE SMOKING STATUS
    fig_count += 1
    ax = fig.add_subplot(gs[-6:, -4:])
    smoks = data_by_subject["smoking_status"].copy()
    perc_annotated = int(
        np.round(
            100 - sum([s == "nan" or pd.isnull(s) for s in smoks]) / len(smoks) * 100,
            0,
        )
    )
    smoks = smoks[smoks != "nan"]
    smoks_freqs = smoks.value_counts()
    n_bars = len(smoks_freqs)
    ax.bar(
        x=np.linspace(0 + 0.5 / n_bars, 1 - 0.5 / n_bars, n_bars),
        tick_label=smoks_freqs.index,
        height=smoks_freqs.values,
        width=barwidth * 5 / 4,
    )
    ax.set_xlim(left=0, right=1)
    print(f"smoking_status: {perc_annotated}% annotated")
    ax.set_ylabel("n subjects", fontsize=fontsize)
    ax.set_xlabel("smoking status", fontsize=fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params("x", rotation=90, labelsize=fontsize)
    ax.grid(False)
    ax.tick_params(bottom=True, left=True, labelsize=fontsize)
    plt.tight_layout()
    plt.grid(False)
    if show:
        plt.show()
    plt.close()
    if return_fig:
        return fig


def plot_subject_and_sample_stats_incl_na(
    adata, fz=10, show_fig=True, return_fig=False, na_color="lightgrey"
):
    with plt.rc_context(
        {
            "figure.figsize": (12, 6),
            "xtick.labelsize": fz,
            "ytick.labelsize": fz,
            "axes.labelsize": fz,
            "font.size": fz,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    ):
        data_by_subject = adata.obs.groupby("subject_ID").agg(
            {
                "age": "first",
                "BMI": "first",
                "ancestry": "first",
                "sex": "first",
                "smoking_status": "first",
            }
        )
        data_by_subject.columns = [
            col.capitalize() if col != "BMI" else col for col in data_by_subject.columns
        ]
        for col in data_by_subject.columns:
            # so that "NA" can be added without having to add new category
            data_by_subject[col] = data_by_subject[col].tolist()
        data_by_subject.fillna("NA", inplace=True)
        data_by_subject.replace("nan", "NA", inplace=True)
        data_by_subject.replace("None", "NA", inplace=True)
        n_rows = 2
        n_cols = 5
        fig = plt.figure()
        #     subplots(
        #         figsize=(figwidth, figheight),
        #     constrained_layout=True,
        #     )
        # gs = GridSpec(12, 12, figure=fig)
        fig_count = 0
        # FIGURE 1 AGE
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)  # gs[:6, :6])
        bins = np.arange(0, max(adata.obs.age), 5)
        tick_idc = np.arange(0, len(bins), 4)
        n_subjects_unannotated = (data_by_subject.Age == "NA").sum()
        perc_annotated = int(
            np.round(
                100 - (n_subjects_unannotated / data_by_subject.shape[0] * 100),
                0,
            )
        )
        ages = data_by_subject.Age[~(data_by_subject.Age == "NA")]
        ax.hist(ages, bins=bins, rwidth=0.9, color="black")
        age_ylim = ax.get_ylim()
        print(f"age: {perc_annotated}% annotated")
        ax.set_xlabel("Age")
        ax.set_xticks(bins[tick_idc])
        ax.tick_params(labelsize=fz, bottom=True, left=True)
        ax.set_ylabel("n subjects")  # , fontsize=fontsize)
        ax.grid(False)
        # FIGURE age unannotated:
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        ax.bar(["NA"], n_subjects_unannotated, width=0.05, color=na_color)
        ax.set_xlim((-0.5, 0.5))
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Age")
        #     ax.tick_params(axis="x", rotation=45)
        ax.set_yticks([])
        ax.set_ylim(age_ylim)
        # FIGURE BMI
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        BMIs = data_by_subject.BMI.copy()
        n_subjects_unannotated = (BMIs == "NA").sum()
        perc_annotated = int(round(100 - (n_subjects_unannotated / len(BMIs) * 100)))
        BMIs = BMIs[~(BMIs == "NA")]
        bins = np.arange(np.floor(BMIs.min() / 2) * 2, BMIs.max(), 2)
        tick_idc = np.arange(0, len(bins), 3)
        ax.hist(BMIs, bins=bins, rwidth=0.9, color="black")
        print(f"BMI: {perc_annotated}% annotated")
        ax.set_xlabel("BMI")
        ax.set_ylabel("n subjects")
        ax.set_xticks(bins[tick_idc])
        ax.tick_params(bottom=True, left=True)
        ax.grid(False)
        bmi_ylim = ax.get_ylim()
        # FIGURE BMI unannotated:
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        ax.bar(["NA"], n_subjects_unannotated, width=0.05, color=na_color)
        ax.set_xlim((-0.5, 0.5))
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("BMI")
        ax.set_yticks([])
        ax.set_ylim(bmi_ylim)
        # FIGURE SEX
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        x_man = np.sum(data_by_subject.Sex == "male")
        x_woman = np.sum(data_by_subject.Sex == "female")
        x_unannotated = np.sum(data_by_subject.Sex == "NA")
        perc_annotated = int(
            np.round(
                100 - x_unannotated / data_by_subject.shape[1] * 100,
                0,
            )
        )
        ax.bar(
            x=[0.1, 0.2, 0.3],
            tick_label=["Male", "Female", "NA"],
            height=[x_man, x_woman, x_unannotated],
            width=0.05,
            color=["black", "black", na_color],
        )
        ax.set_xlim(left=0, right=1)
        print(f"sex: {perc_annotated}% annotated)")
        ax.tick_params("x", rotation=90, bottom=True, left=True)
        ax.tick_params("y", bottom=True, left=True)
        ax.set_ylabel("n subjects")
        ax.set_xlabel("Sex")
        ax.grid(False)
        # FIGURE 4 ANCESTRY
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        ethns = data_by_subject.Ancestry.copy()
        n_subjects_unannotated = (ethns == "NA").sum()
        perc_annotated = int(
            np.round(100 - n_subjects_unannotated / len(ethns) * 100, 0)
        )
        ethn_freqs = ethns.value_counts()
        ethn_freqs.index = [
            idx.capitalize() if idx != "NA" else idx for idx in ethn_freqs.index
        ]
        n_bars = len(ethn_freqs)
        # move unannoated to the right side
        ethn_freqs = ethn_freqs[
            [idx for idx in ethn_freqs.index if idx != "NA"] + ["NA"]
        ]
        ax.bar(
            x=np.linspace(0.075, n_bars * 0.075 + 0.075, n_bars),
            tick_label=ethn_freqs.index,
            height=ethn_freqs.values,
            width=0.05,
            color=(len(ethn_freqs) - 1) * ["black"] + [na_color],
        )
        ax.set_xlim(left=0, right=1)
        print(f"ancestry {perc_annotated}% annotated")
        ax.set_ylabel("n subjects")
        ax.set_xlabel("Ancestry")
        ax.tick_params("x", rotation=90, bottom=True)
        ax.tick_params("y", left=True)
        ax.grid(False)
        # FIGURE SMOKING STATUS
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        smoks = data_by_subject.Smoking_status.copy()
        n_subjects_unannotated = (smoks == "NA").sum()
        perc_annotated = int(
            np.round(
                100 - n_subjects_unannotated / len(smoks) * 100,
                0,
            )
        )
        smoks_freqs = smoks.value_counts()
        # move unannoated to the right side
        smoks_freqs = smoks_freqs[
            [idx for idx in smoks_freqs.index if idx != "NA"] + ["NA"]
        ]
        smoks_freqs.index = [
            idx.capitalize() if idx != "NA" else idx for idx in smoks_freqs.index
        ]
        n_bars = len(smoks_freqs)
        ax.bar(
            x=[0.1, 0.2, 0.3, 0.4],
            tick_label=smoks_freqs.index,
            height=smoks_freqs.values,
            width=0.05,
            color=(len(smoks_freqs) - 1) * ["black"] + [na_color],
        )
        ax.set_xlim(left=0, right=1)
        print(f"smoking_status: {perc_annotated}% annotated")
        ax.set_ylabel("n subjects")
        ax.set_xlabel("Smoking status")
        ax.tick_params("x", rotation=90)
        ax.grid(False)
        ax.tick_params(bottom=True, left=True)
        plt.tight_layout()
        plt.grid(False)
        # FIGURE ANATOMICAL REGION CCF SCORE
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        data_by_sample = adata.obs.groupby("sample").agg(
            {"anatomical_region_ccf_score": "first"}
        )
        data_by_sample.columns = [
            col.capitalize().replace("_", " ") for col in data_by_sample.columns
        ]
        data_by_sample.fillna("NA", inplace=True)
        data_by_sample.replace("nan", "NA", inplace=True)
        data_by_sample.replace("None", "NA", inplace=True)
        regs = data_by_sample["Anatomical region ccf score"].copy()
        n_subjects_unannotated = (regs == "NA").sum()
        perc_annotated = int(
            np.round(
                100 - n_subjects_unannotated / len(regs) * 100,
                0,
            )
        )
        regs = regs[regs != "NA"]
        ax.hist(regs, color="black", width=0.05)
        ax.set_ylabel("n samples")
        ax.set_xlabel("Anatomical region\nccf score")
        ax.tick_params(bottom=True, left=True)
        ccf_ylim = ax.get_ylim()
        ax.grid(False)
        # FIGURE BMI unannotated:
        fig_count += 1
        ax = fig.add_subplot(n_rows, n_cols, fig_count)
        ax.bar(["NA"], n_subjects_unannotated, width=0.05, color=na_color)
        ax.set_xlim((-0.5, 0.5))
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Anatomical region\nccf score")
        ax.set_yticks([])
        ax.set_ylim(ccf_ylim)
        # show and return
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.close()
        if return_fig:
            return fig
