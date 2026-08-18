#!/usr/bin/env python3
"""
Sankey Diagram Visualization for Cell Type Mapping

This module provides Sankey diagram visualization for comparing cell type
annotations between different methods or datasets.

Reference: Original script by Nikolay Markov
https://github.com/NUPulmonary/scarches-covid-reference/blob/master/sankey.py
Adapted by Lisa Sikkema for HLCA reproducibility

Usage:
    from sankey import sankey

    # Create Sankey diagram
    fig = sankey(
        x=original_labels,
        y=predicted_labels,
        colorside="left",
        title="Cell Type Comparison"
    )
    fig.savefig("sankey.pdf")
"""

import collections
import colorsys

import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot


def get_distinct_colors(n):
    """
    Generate n visually distinct RGB colors.

    Reference: https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python

    Parameters
    ----------
    n : int
        Number of colors to generate

    Returns
    -------
    list
        List of RGB tuples
    """
    hue_partition = 1 / (n + 1)
    colors = [
        colorsys.hsv_to_rgb(hue_partition * value, 1.0, 1.0) for value in range(n)
    ]
    return colors[::2] + colors[1::2]


def text_width(fig, ax, text, fontsize):
    """Calculate the width of text in inches."""
    text = ax.text(-100, 0, text, fontsize=fontsize)
    text_bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    text_bb = text_bb.transformed(fig.dpi_scale_trans.inverted())
    width = text_bb.width
    text.remove()
    return width


class Sankey:
    """
    Sankey diagram class for visualizing flow between two categorical variables.

    Parameters
    ----------
    x : array-like
        Left-side categories (e.g., original labels)
    y : array-like
        Right-side categories (e.g., predicted labels)
    colorside : str
        Which side to use for coloring flows ("left" or "right")
    plot_width : float
        Width of the plot in inches
    plot_height : float
        Height of the plot in inches
    gap : float
        Gap between nodes
    alpha : float
        Transparency of flows
    fontsize : str or int
        Font size for labels
    left_order : list, optional
        Order of left-side categories
    mapping : object, optional
        Mapping object for category colors
    colors : dict, optional
        Custom color mapping
    tag : str, optional
        Tag to display on the plot
    title : str, optional
        Main title
    title_left : str, optional
        Title for left side
    title_right : str, optional
        Title for right side
    ax : matplotlib.axes.Axes, optional
        Existing axes to use
    """

    def __init__(
        self,
        x,
        y,
        colorside,
        plot_width=8,
        plot_height=8,
        gap=0.12,
        alpha=0.3,
        fontsize="small",
        left_order=None,
        mapping=None,
        colors=None,
        tag=None,
        title=None,
        title_left=None,
        title_right=None,
        ax=None,
    ):
        self.X = x
        self.Y = y
        if ax:
            self.plot_width = ax.get_position().width * ax.figure.get_size_inches()[0]
            self.plot_height = ax.get_position().height * ax.figure.get_size_inches()[1]
        else:
            self.plot_width = plot_width
            self.plot_height = plot_height
        self.gap = gap
        self.alpha = alpha
        self.colors = colors
        self.colorside = colorside
        self.fontsize = fontsize
        self.tag = tag
        self.map = mapping is not None
        self.mapping = mapping
        self.mapping_colors = {
            "increase": "#1f721c",
            "decrease": "#ddc90f",
            "mistake": "#dd1616",
            "correct": "#dddddd",
            "novel": "#59a8d6",
        }
        self.title = title
        self.title_left = title_left
        self.title_right = title_right

        self.need_title = any(
            value is not None for value in (title, title_left, title_right)
        )
        if self.need_title:
            self.plot_height -= 0.5

        self.init_figure(ax)

        self.flows = collections.Counter(zip(x, y))
        self.init_nodes(left_order)

        self.init_widths()
        # inches per 1 item in x and y
        self.resolution = (plot_height - gap * (len(self.left_nodes) - 1)) / len(x)
        if self.colors is None:
            if colorside == "left":
                self.colors = {
                    name: colour
                    for name, colour in zip(
                        self.left_nodes.keys(),
                        get_distinct_colors(len(self.left_nodes)),
                    )
                }
            elif colorside == "right":
                self.colors = {
                    name: colour
                    for name, colour in zip(
                        self.right_nodes.keys(),
                        get_distinct_colors(len(self.right_nodes)),
                    )
                }
            else:
                raise ValueError(
                    "colorside argument should be set either to 'left' or 'right'. Exiting."
                )

        self.init_offsets()

    def init_figure(self, ax):
        if ax is None:
            self.fig = matplotlib.pyplot.figure()
            self.ax = matplotlib.pyplot.Axes(self.fig, [0, 0, 1, 1])
            self.fig.add_axes(self.ax)
        self.fig = ax.figure
        self.ax = ax

    def init_nodes(self, left_order):
        left_nodes = {}
        right_nodes = {}
        for (left, right), flow in self.flows.items():
            if left in left_nodes:
                left_nodes[left] += flow
            else:
                left_nodes[left] = flow
            if right in right_nodes:
                node = right_nodes[right]
                node[0] += flow
                if flow > node[2]:
                    node[1] = left
                    node[2] = flow
            else:
                right_nodes[right] = [flow, left, flow]

        self.left_nodes = collections.OrderedDict()
        self.left_nodes_idx = {}
        if left_order is None:
            key = lambda pair: -pair[1]
        else:
            left_order = list(left_order)
            key = lambda pair: left_order.index(pair[0])

        for name, flow in sorted(left_nodes.items(), key=key):
            self.left_nodes[name] = flow
            self.left_nodes_idx[name] = len(self.left_nodes_idx)

        left_names = list(self.left_nodes.keys())
        self.right_nodes = collections.OrderedDict()
        self.right_nodes_idx = {}
        for name, node in sorted(
            right_nodes.items(),
            key=lambda pair: (left_names.index(pair[1][1]), -pair[1][2]),
        ):
            self.right_nodes[name] = node[0]
            self.right_nodes_idx[name] = len(self.right_nodes_idx)

    def init_widths(self):
        self.left_width = max(
            text_width(self.fig, self.ax, node, self.fontsize)
            for node in self.left_nodes
        )
        if self.title_left:
            self.left_width = max(
                self.left_width,
                text_width(self.fig, self.ax, self.title_left, self.fontsize) / 2,
            )
        self.right_width = max(
            text_width(self.fig, self.ax, node, self.fontsize)
            for node in self.right_nodes
        )
        if self.title_right:
            self.right_width = max(
                self.right_width,
                text_width(self.fig, self.ax, self.title_right, self.fontsize) / 2,
            )

        self.right_stop = self.plot_width - self.left_width - self.right_width
        self.middle1_stop = self.right_stop * 9 / 20
        self.middle2_stop = self.right_stop * 11 / 20

    def init_offsets(self):
        self.offsets_l = {}
        self.offsets_r = {}

        offset = 0
        for name, flow in self.left_nodes.items():
            self.offsets_l[name] = offset
            offset += flow * self.resolution + self.gap

        offset = 0
        for name, flow in self.right_nodes.items():
            self.offsets_r[name] = offset
            offset += flow * self.resolution + self.gap

    def draw_flow(self, left, right, flow, node_offsets_l, node_offsets_r, colorside):
        P = matplotlib.path.Path

        flow *= self.resolution
        left_y = self.offsets_l[left] + node_offsets_l[left]
        right_y = self.offsets_r[right] + node_offsets_r[right]
        if self.need_title:
            left_y += 0.5
            right_y += 0.5
        node_offsets_l[left] += flow
        node_offsets_r[right] += flow
        if colorside == "left":
            color = self.colors[left]
        elif colorside == "right":
            color = self.colors[right]
        if self.mapping is not None:
            color = self.mapping_colors[self.mapping.category(left, right)]

        path_data = [
            (P.MOVETO, (0, -left_y)),
            (P.LINETO, (0, -left_y - flow)),
            (P.CURVE4, (self.middle1_stop, -left_y - flow)),
            (P.CURVE4, (self.middle2_stop, -right_y - flow)),
            (P.CURVE4, (self.right_stop, -right_y - flow)),
            (P.LINETO, (self.right_stop, -right_y)),
            (P.CURVE4, (self.middle2_stop, -right_y)),
            (P.CURVE4, (self.middle1_stop, -left_y)),
            (P.CURVE4, (0, -left_y)),
            (P.CLOSEPOLY, (0, -left_y)),
        ]
        codes, verts = zip(*path_data)
        path = P(verts, codes)
        patch = matplotlib.patches.PathPatch(
            path,
            facecolor=color,
            alpha=0.9 if flow < 0.02 else self.alpha,
            edgecolor="none",
        )
        self.ax.add_patch(patch)

    def draw_label(self, label, is_left):
        nodes = self.left_nodes if is_left else self.right_nodes
        offsets = self.offsets_l if is_left else self.offsets_r
        y = offsets[label] + nodes[label] * self.resolution / 2
        if self.need_title:
            y += 0.5

        self.ax.text(
            -0.1 if is_left else self.right_stop + 0.1,
            -y,
            label,
            horizontalalignment="right" if is_left else "left",
            verticalalignment="center",
            fontsize=self.fontsize,
        )

    def draw_titles(self):
        if self.title:
            self.ax.text(
                self.right_stop / 2,
                -0.25,
                self.title,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=self.fontsize,
                fontweight="bold",
            )
        if self.title_left:
            self.ax.text(
                -0.1,
                -0.25,
                self.title_left,
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=self.fontsize,
            )
        if self.title_right:
            self.ax.text(
                self.right_stop + 0.1,
                -0.25,
                self.title_right,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=self.fontsize,
            )

    def draw(self, colorside):
        node_offsets_l = collections.Counter()
        node_offsets_r = collections.Counter()

        for (left, right), flow in sorted(
            self.flows.items(),
            key=lambda pair: (
                self.left_nodes_idx[pair[0][0]],
                self.right_nodes_idx[pair[0][1]],
            ),
        ):
            self.draw_flow(left, right, flow, node_offsets_l, node_offsets_r, colorside)

        for name in self.left_nodes:
            self.draw_label(name, True)
        for name in self.right_nodes:
            self.draw_label(name, False)
        self.draw_titles()

        self.ax.axis("equal")
        self.ax.set_xlim(
            -self.left_width - self.gap, self.right_stop + self.gap + self.right_width
        )
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        for k in self.ax.spines:
            self.ax.spines[k].set_visible(False)

        if self.tag:
            text_ax = self.fig.add_axes((0.02, 0.95, 0.05, 0.05), frame_on=False)
            text_ax.set_axis_off()
            matplotlib.pyplot.text(
                0, 0, self.tag, fontsize=30, transform=text_ax.transAxes
            )


def sankey(x, y, colorside="left", **kwargs):
    """
    Create a Sankey diagram comparing two categorical variables.

    Parameters
    ----------
    x : array-like
        Left-side categories (e.g., original labels)
    y : array-like
        Right-side categories (e.g., predicted labels)
    colorside : str
        Which side to use for coloring flows ("left" or "right")
    **kwargs
        Additional arguments passed to Sankey class

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the Sankey diagram

    Examples
    --------
    >>> import pandas as pd
    >>> # Compare original vs predicted cell types
    >>> original = ["T cell", "T cell", "B cell", "Macrophage", "B cell"]
    >>> predicted = ["T cell", "NK cell", "B cell", "Monocyte", "B cell"]
    >>> fig = sankey(original, predicted, colorside="left", title="Comparison")
    >>> fig.savefig("comparison.pdf")
    """
    diag = Sankey(x, y, colorside, **kwargs)
    diag.draw(colorside)
    return diag.fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Sankey diagram from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create Sankey from CSV with two columns
    python sankey.py --input comparison.csv --col-left original --col-right predicted --output sankey.pdf

Requirements:
    pip install matplotlib pandas
        """,
    )
    parser.add_argument(
        "--input", "-i", help="Path to input CSV file with two categorical columns"
    )
    parser.add_argument(
        "--col-left",
        default="original",
        help="Column name for left side (default: original)",
    )
    parser.add_argument(
        "--col-right",
        default="predicted",
        help="Column name for right side (default: predicted)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sankey.pdf",
        help="Output file path (default: sankey.pdf)",
    )
    parser.add_argument(
        "--colorside",
        default="left",
        choices=["left", "right"],
        help="Which side to color by (default: left)",
    )
    parser.add_argument("--title", default=None, help="Plot title")

    args = parser.parse_args()

    if args.input:
        import pandas as pd

        df = pd.read_csv(args.input)
        fig = sankey(
            df[args.col_left].tolist(),
            df[args.col_right].tolist(),
            colorside=args.colorside,
            title=args.title,
        )
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        # Demo
        print("Running demo...")
        import matplotlib.pyplot as plt

        original = ["T cell"] * 100 + ["B cell"] * 80 + ["Macrophage"] * 50
        predicted = (
            ["T cell"] * 90
            + ["NK cell"] * 10
            + ["B cell"] * 75
            + ["Plasma cell"] * 5
            + ["Macrophage"] * 40
            + ["Monocyte"] * 10
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        diag = Sankey(
            original,
            predicted,
            colorside="left",
            title="Cell Type Comparison Demo",
            title_left="Original",
            title_right="Predicted",
            ax=ax,
        )
        diag.draw("left")
        fig.savefig("sankey_demo.pdf", bbox_inches="tight")
        print("Saved: sankey_demo.pdf")
