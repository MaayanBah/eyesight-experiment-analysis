from __future__ import annotations
import csv
import math
import os.path
import statistics
from enum import Enum
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import stats
from AnalyzedExperiments import AnalyzedExperiments, limit_standard_deviation
from itertools import zip_longest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from experiment_types import Eyesight, ScreenLocation
from configurations import (MAX_DEVIATION, RED, DARK_RED, GREEN, DARK_GREEN, BLUE, LIGHT_GREY,
                            LIGHT_BLUE, LIGHT_PURPLE, SAMPLE_RATE, DURATION)
from scipy.fft import fft, fftfreq


class GraphType(Enum):
    Scattered = 1
    Line = 2


def create_bar_graph(categories: list[str],
                     values: list[float | int],
                     x_label: str,
                     y_label: str,
                     graph_title: str,
                     y_errors: list[float | int] = None,
                     bar_width=0.4) -> tuple[plt.figure, any]:
    fig, ax = plt.subplots()
    ax.bar(categories, values, width=bar_width, align='center', color=BLUE, yerr=y_errors, capsize=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(graph_title)
    return fig, ax


def create_time_series_scattered_or_line_graph_sorted_by_time(group_name_to_locations: dict[str, list[float]],
                                                              group_name_to_color: dict[str, str],
                                                              x_label: str,
                                                              y_label: str,
                                                              title: str,
                                                              graph_type: GraphType,
                                                              legend_group_names: list[str],
                                                              dot_size: int = 20,
                                                              line_width: float = 1,
                                                              add_fill_between: bool = False,
                                                              figsize=(20, 6),
                                                              fill_between_values: dict[str, list[float]] = None
                                                              ) -> tuple[plt.figure, any]:
    """
    :param fill_between_values:
    :param figsize:
    :param add_fill_between:
    :param legend_group_names:
    :param line_width:
    :param dot_size: The dots size.
    :param group_name_to_locations: group name to it's values sorted by time, if for a certain time there's no data
                                fill it with None - all groups must have the same list length.
                                The same group can have several lists.
    :param group_name_to_color: A dictionary from each group name to it's color (RGB or known color names).
    :param x_label: The X axis title.
    :param y_label: The Y axis title.
    :param title: The graph title.
    :param graph_type:
    :return: The new figure and the ax_dict (for more info look at matplotlib.subplots documentation)
    """
    fig, ax = plt.subplots(figsize=figsize)
    try_handles = []
    try_labels = []
    for group_name, locations in group_name_to_locations.items():
        locations = [np.nan if loc is None else loc for loc in locations]
        x_values = [x for x in range(1, len(locations) + 1)]
        # Create a scatter plot for each group
        if graph_type == GraphType.Scattered:
            cur_plot: plt.figure = ax.scatter(
                x_values,
                locations, color=group_name_to_color[group_name],
                label=group_name, s=dot_size
            )
        else:
            cur_plot: plt.figure = ax.plot(
                x_values,
                locations, color=group_name_to_color[group_name],
                label=group_name,
                linewidth=line_width,
                zorder=1
            )
            if add_fill_between and group_name in fill_between_values.keys():
                lower_bound = locations - np.array(fill_between_values[group_name])
                upper_bound = locations + np.array(fill_between_values[group_name])
                ax.fill_between(
                    x_values,
                    lower_bound,
                    upper_bound,
                    color=group_name_to_color[group_name],
                    alpha=0.3,
                    zorder=2
                )
        if group_name in legend_group_names:
            try_handles.append(cur_plot)
            try_labels.append(group_name)

        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    if graph_type == GraphType.Line:
        # scattered and line plot return different type of objects, so we reformat the "Line" for the legend.
        try_handles = [item for sublist in try_handles for item in sublist]

    # handles, labels = ax.get_legend_handles_labels()
    if legend_group_names:
        ax.legend(handles=try_handles, labels=try_labels)

    return fig, ax


def create_scattered_graph(group_name_to_locations: dict[str, list[ScreenLocation]],
                           group_name_to_color: dict[str, str],
                           x_label: str,
                           y_label: str,
                           title: str,
                           dot_size: int = 20,
                           create_legend=True,
                           add_middle_line=False,
                           add_x_equal_y_line=False) -> tuple[plt.figure, any]:
    """
    :param add_x_equal_y_line:
    :param add_middle_line:
    :param create_legend:
    :param dot_size:
    :param group_name_to_locations: A dictionary from group name to a list of x, y indexes tuple.
    :param group_name_to_color: A dictionary from group name to the color of it in the plot.
    :param x_label: The X axis title.
    :param y_label: The Y axis title.
    :param title: The graph title.
    :return:
    """
    fig, ax = plt.subplots()
    all_x_coords = []
    all_y_coords = []

    for group_name, locations in group_name_to_locations.items():
        x_coords = [location.x for location in locations]
        y_coords = [location.y for location in locations]

        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)

        # Create a scatter plot for each group
        ax.scatter(x_coords,
                   y_coords,
                   color=group_name_to_color[group_name],
                   label=group_name, s=dot_size)

        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if add_middle_line:
            coeffs = np.polyfit(x_coords, y_coords, 1)
            slope, intercept = coeffs
            x_vals = np.array(ax.get_xlim())
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color=group_name_to_color[group_name], linestyle='-', linewidth=1)

    if add_x_equal_y_line:
        # Add a y = x line
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        min_val = max(min(x_limits[0], y_limits[0]), min(x_limits[1], y_limits[1]))
        max_val = min(max(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1]))
        ax.plot([min_val, max_val], [min_val, max_val], color=LIGHT_GREY, linestyle='--', linewidth=1)

    if create_legend:
        ax.legend()

    ax.set_title(title)
    return fig, ax


def create_scattered_k_means_graph(locations_list,
                                   n_clusters,
                                   x_label: str,
                                   y_label: str,
                                   title: str) -> tuple[plt.figure, any]:
    fig, ax = plt.subplots()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(locations_list)

    ax.scatter(
        [x_y[0] for x_y in locations_list],
        [x_y[1] for x_y in locations_list],
        c=kmeans.labels_,
        label='Clusters')

    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return fig, ax


def matplotlib_figures_to_pdf(pdf_path: str, *figures: plt.figure) -> None:
    """
    :param pdf_path: The path of the PDF file.
    :param figures: Figure of the graphs you want to present in the PDF file.
    :return: None, but creates a PDF with the given figures.
    """
    tries = 0
    while tries < 3:
        output_full_path = f"{pdf_path}.pdf" if tries == 0 else f"{pdf_path}_{tries}.pdf"
        try:
            pdf_pages: PdfPages = PdfPages(output_full_path)
            for fig in figures:
                pdf_pages.savefig(fig)
            pdf_pages.close()
            return
        except OSError:
            tries += 1
            print(f"The file '{output_full_path}' cannot be created or accessed, creating the file under a different"
                  f" name: {pdf_path}_{tries}")
    raise OSError(f"Error: The file '{pdf_path}' cannot be created or accessed. It may be open in another application.")


def create_graphs_of_good_vs_bad_eyesight_fixation_data(
        good_analyzed_experiments: AnalyzedExperiments,
        bad_analyzed_experiments: AnalyzedExperiments,
        ids_order) -> tuple[plt.figure, ...]:
    def get_data_for_graphs(analyzed_experiments: AnalyzedExperiments):
        real_id_to_num_fixations: dict[str, int] = {}
        real_id_to_duration_mean: dict[str, int] = {}
        for analyzed_experiment in analyzed_experiments.analyzed_experiments.values():
            num_fixations, duration_mean = analyzed_experiment.get_num_of_fixation_and_mean_duration_in_video()
            real_id_to_num_fixations[analyzed_experiment.experiment.experiment_real_id] = num_fixations
            real_id_to_duration_mean[analyzed_experiment.experiment.experiment_real_id] = duration_mean
        (num_of_fixation_mean,
         duration_mean,
         fixations_sem,
         duration_sem) = analyzed_experiments.get_mean_fixations_count_and_duration()
        return (real_id_to_num_fixations, real_id_to_duration_mean, num_of_fixation_mean, duration_mean,
                fixations_sem, duration_sem)

    """
    :param bad_analyzed_experiments: AnalyzedExperiments class of experiment of people with good eyesight.
    :param good_analyzed_experiments: AnalyzedExperiments class of experiment of people with bad eyesight
    :return: A graph representing the mean number of fixations and the mean duration of it, divided by
             good and bad eyesight.
    """
    (good_real_id_to_num_fixations,
     good_real_id_to_duration_mean,
     good_num_of_fixation_mean,
     good_duration_mean,
     good_fixations_sem,
     good_duration_sem) = get_data_for_graphs(good_analyzed_experiments)

    (bad_real_id_to_num_fixations,
     bad_real_id_to_duration_mean,
     bad_num_of_fixation_mean,
     bad_duration_mean,
     bad_fixations_sem,
     bad_duration_sem) = get_data_for_graphs(bad_analyzed_experiments)

    fixation_differences = {
        experiment_real_id: number_of_good_fixations - bad_real_id_to_num_fixations[experiment_real_id]
        for experiment_real_id, number_of_good_fixations in good_real_id_to_num_fixations.items()
    }

    duration_mean_differences = {
        experiment_real_id: good_duration_mean - bad_real_id_to_duration_mean[experiment_real_id]
        for experiment_real_id, good_duration_mean in good_real_id_to_duration_mean.items()
    }

    t_stat_num_fixations, p_value_num_fixations = stats.ttest_rel(
        [bad_real_id_to_num_fixations[id_] for id_ in ids_order],
        [good_real_id_to_num_fixations[id_] for id_ in ids_order]
    )
    correlation_num_fixations = np.corrcoef(
        [bad_real_id_to_num_fixations[id_] for id_ in ids_order],
        [good_real_id_to_num_fixations[id_] for id_ in ids_order]
    )[0, 1]

    fixations_fig, _ = create_bar_graph(
        [str(Eyesight.GOOD), str(Eyesight.BAD)],
        [good_num_of_fixation_mean, bad_num_of_fixation_mean],
        "Eyesight",
        "Number of Fixations (Average)",
        "Average Number of Fixations\n"
        f"t-statistic: {t_stat_num_fixations:.3f},"
        f" p-value: {p_value_num_fixations:.3f},",
        y_errors=[good_fixations_sem, bad_fixations_sem]
    )

    single_experiments_num_fixations, _ = create_scattered_graph(
        {
            str(Eyesight.GOOD): [
                ScreenLocation(int(experiment_real_id), number_of_fixations)
                for experiment_real_id, number_of_fixations in good_real_id_to_num_fixations.items()
            ],
            str(Eyesight.BAD): [
                ScreenLocation(int(experiment_real_id), number_of_fixations)
                for experiment_real_id, number_of_fixations in bad_real_id_to_num_fixations.items()
            ]
        },
        {
            str(Eyesight.GOOD): GREEN,
            str(Eyesight.BAD): RED
        },
        "Participant",
        "Number of Fixations",
        "Number of Fixations per Experiment"

    )

    fixation_differences_fig, _ = create_scattered_graph(
        {
            "Differences per Participant": [
                ScreenLocation(int(experiment_real_id), fixation_differences)
                for experiment_real_id, fixation_differences in fixation_differences.items()
            ]
        },
        {
            "Differences per Participant": LIGHT_BLUE
        },
        "Participant",
        "Fixations Difference",
        "Fixations Difference per Experiment",
    )

    num_fixations_y_good_x_bad: list[ScreenLocation] = [
        ScreenLocation(
            bad_real_id_to_num_fixations[experiment_real_id],
            good_real_id_to_num_fixations[experiment_real_id])
        for experiment_real_id in ids_order
    ]

    single_experiments_num_fixations_y_good_x_bad, _ = create_scattered_graph(
        {
            "Participants": num_fixations_y_good_x_bad
        },
        {
            "Participants": LIGHT_PURPLE
        },
        "Bad Eyesight Number of Fixations",
        "Good Eyesight Number of Fixations",
        f"Number of Fixations per Participant\n"
        # f"t-statistic: {t_stat_num_fixations:.3f},"
        # f" p-value: {p_value_num_fixations:.3f},"
        f" correlation: {correlation_num_fixations:.3f}",
        add_middle_line=True,
        add_x_equal_y_line=True)

    t_stat_duration_mean, p_value_duration_mean = stats.ttest_rel(
        [bad_real_id_to_duration_mean[id_] for id_ in ids_order],
        [good_real_id_to_duration_mean[id_] for id_ in ids_order]
    )

    correlation_duration_mean = np.corrcoef(
        [bad_real_id_to_duration_mean[id_] for id_ in ids_order],
        [good_real_id_to_duration_mean[id_] for id_ in ids_order]
    )[0, 1]

    duration_fig, _ = create_bar_graph(
        [str(Eyesight.GOOD), str(Eyesight.BAD)],
        [good_duration_mean, bad_duration_mean],
        "Eyesight",
        "Average Duration",
        "Average Fixation Duration [ms]\n"
        f"t-statistic: {t_stat_duration_mean:.3f},"
        f" p-value: {p_value_duration_mean:.3f},",
        # f" correlation: {correlation_duration_mean:.3f}",
        y_errors=[good_duration_sem, bad_duration_sem]
    )

    single_experiments_duration_mean, _ = create_scattered_graph(
        {
            str(Eyesight.GOOD): [
                ScreenLocation(int(experiment_real_id), duration_mean)
                for experiment_real_id, duration_mean in good_real_id_to_duration_mean.items()
            ],
            str(Eyesight.BAD): [
                ScreenLocation(int(experiment_real_id), duration_mean)
                for experiment_real_id, duration_mean in bad_real_id_to_duration_mean.items()
            ]
        },
        {
            str(Eyesight.GOOD): GREEN,
            str(Eyesight.BAD): RED
        },
        "Participant",
        "Duration Mean",
        "Duration Mean per Participant"
    )

    duration_mean_differences_fig, _ = create_scattered_graph(
        {
            "Differences per experiment": [
                ScreenLocation(int(experiment_real_id), duration_mean_differences)
                for experiment_real_id, duration_mean_differences in duration_mean_differences.items()
            ]
        },
        {
            "Differences per experiment": LIGHT_BLUE
        },
        "Participant",
        "Duration Mean Difference",
        "Duration Mean Difference per Participant",
    )

    duration_mean_y_good_x_bad: list[ScreenLocation] = [
        ScreenLocation(
            bad_real_id_to_duration_mean[experiment_real_id],
            good_real_id_to_duration_mean[experiment_real_id]
        )
        for experiment_real_id in ids_order
    ]
    single_experiments_duration_mean_y_good_x_bad, _ = create_scattered_graph(
        {
            "Participants": duration_mean_y_good_x_bad
        },
        {
            "Participants": LIGHT_PURPLE
        },
        "Bad Eyesight Duration Mean",
        "Good Eyesight Duration Mean",
        f"Duration Mean per Participant\n"
        # f"t-statistic: {t_stat_duration_mean:.3f},"
        # f" p-value: {p_value_duration_mean:.3f}"
        f"correlation: {correlation_duration_mean:.3f}",
        add_middle_line=True,
        add_x_equal_y_line=True
    )

    return (fixations_fig,
            duration_fig,
            single_experiments_num_fixations,
            fixation_differences_fig,
            single_experiments_num_fixations_y_good_x_bad,
            num_fixations_y_good_x_bad,
            single_experiments_duration_mean,
            duration_mean_differences_fig,
            single_experiments_duration_mean_y_good_x_bad,
            duration_mean_y_good_x_bad)


def create_fixations_count_and_duration_k_means_graph(
        good_analyzed_experiments: AnalyzedExperiments,
        bad_analyzed_experiments: AnalyzedExperiments) -> tuple[plt.figure, plt.figure]:
    """
    :param bad_analyzed_experiments: AnalyzedExperiments class of experiment of people with good eyesight.
    :param good_analyzed_experiments: AnalyzedExperiments class of experiment of people with bad eyesight
    :return: A graph representing the
    """
    (good_list_of_fixation_count,
     good_duration_mean_list) = good_analyzed_experiments.get_list_of_mean_fixations_count_and_duration_per_experiment()
    good_fixations_durations: list[ScreenLocation] = [
        ScreenLocation(x, y) for x, y in list(zip(good_list_of_fixation_count, good_duration_mean_list))
    ]

    (bad_list_of_fixation_count,
     bad_duration_mean_list) = bad_analyzed_experiments.get_list_of_mean_fixations_count_and_duration_per_experiment()
    bad_fixations_durations: list[ScreenLocation] = [
        ScreenLocation(x, y) for x, y in list(zip(bad_list_of_fixation_count, bad_duration_mean_list))
    ]

    fixations_counts = good_list_of_fixation_count + bad_list_of_fixation_count
    duration_means = good_duration_mean_list + bad_duration_mean_list

    fixations_count_and_duration_divided_to_good_bad_graph, _ = create_scattered_graph(
        {str(Eyesight.GOOD): good_fixations_durations, str(Eyesight.BAD): bad_fixations_durations},
        {str(Eyesight.GOOD): GREEN, str(Eyesight.BAD): RED},
        "Fixation Count",
        "Fixation Duration",
        "Fixation Count vs Fixation Duration for each experiment")

    fixations_count_and_duration_k_means_graph, _ = create_scattered_k_means_graph(
        list(zip(fixations_counts, duration_means)),
        2,
        "Fixation Count",
        "Fixation Duration",
        "Fixation Count vs Fixation Duration - K-Means"
    )

    return fixations_count_and_duration_divided_to_good_bad_graph, fixations_count_and_duration_k_means_graph


def calculate_mean_of_lists_values(num_lists_: list[list[float]]) -> list[float | None]:
    mean_list: list[float | None] = [
        (sum(val for val in col if val is not None) / len([val for val in col if val is not None]))
        if any(val is not None for val in col) and len([val for val in col if val is not None]) > 0
        else None
        for col in zip_longest(*num_lists_, fillvalue=None)
    ]
    return mean_list


def get_gaze_variance_graphs(good_analyzed_experiments: AnalyzedExperiments,
                             bad_analyzed_experiments: AnalyzedExperiments) -> tuple[plt.figure, plt.figure]:
    good_experiments_id_to_color = {
        experiment_id: GREEN for experiment_id
        in good_analyzed_experiments.experiment_gaze_variances_sorted_by_time.keys()
    }
    bad_experiments_id_to_color = {
        experiment_id: RED for experiment_id
        in bad_analyzed_experiments.experiment_gaze_variances_sorted_by_time.keys()
    }

    fig_variance, ax = create_time_series_scattered_or_line_graph_sorted_by_time(
        {**good_analyzed_experiments.experiment_gaze_variances_sorted_by_time,
         **bad_analyzed_experiments.experiment_gaze_variances_sorted_by_time},
        {**good_experiments_id_to_color, **bad_experiments_id_to_color},
        "Time",
        "Gaze Variance",
        "Variance (Screen Coordinates)",
        GraphType.Scattered,
        [],
        3
    )

    # create scatter plot of all the experiment's mean variance.
    good_variances_means: list[float | None] = calculate_mean_of_lists_values(
        [variances for variances
         in good_analyzed_experiments.experiment_gaze_variances_sorted_by_time.values()]
    )

    bad_variances_means: list[float | None] = calculate_mean_of_lists_values(
        [variances for variances
         in bad_analyzed_experiments.experiment_gaze_variances_sorted_by_time.values()]
    )

    fig_variance_mean, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {str(Eyesight.GOOD): good_variances_means, str(Eyesight.BAD): bad_variances_means},
        {str(Eyesight.GOOD): GREEN, str(Eyesight.BAD): RED},
        "Time Passed",
        "Gaze Variance (Average)",
        "Average Gaze Variance (Screen Coordinates)",
        GraphType.Scattered,
        [str(Eyesight.GOOD), str(Eyesight.BAD)]
    )

    return fig_variance, fig_variance_mean


def get_fixations_number_graphs(good_analyzed_experiments: AnalyzedExperiments,
                                bad_analyzed_experiments: AnalyzedExperiments) -> tuple[plt.figure, plt.figure]:
    def get_group_data_for_graphs(
            analyzed_experiments: AnalyzedExperiments) -> tuple[list[int | float], list[int | float], list[float]]:
        eyesight_fixation_count_sorted_by_time: list[list[int]] = (
            analyzed_experiments.fixation_count_sorted_by_time
        )
        eyesight_fixation_count_sorted_by_time_limited_stdev: list[list[int]] = [
            limit_standard_deviation(fixation_count, max_deviation=MAX_DEVIATION)
            for fixation_count in eyesight_fixation_count_sorted_by_time
        ]
        eyesight_stdev: list[int | float] = [
            statistics.stdev(x) for x in eyesight_fixation_count_sorted_by_time_limited_stdev
        ]

        eyesight_sem: list[float] = [
            std / math.sqrt(len(x)) for std, x in
            zip(eyesight_stdev, eyesight_fixation_count_sorted_by_time_limited_stdev)
        ]

        eyesight_fixation_average_sorted_by_time = [
            sum(fixation_count) / len(fixation_count) for fixation_count
            in eyesight_fixation_count_sorted_by_time_limited_stdev
        ]
        return (eyesight_stdev,
                eyesight_sem,
                eyesight_fixation_average_sorted_by_time)

    """
    :param good_analyzed_experiments: AnalyzedExperiments class of experiment of people with bad eyesight
    :param bad_analyzed_experiments: AnalyzedExperiments class of experiment of people with good eyesight.
     :return: Two graphs:
        1. Fixation Mean Over Time: This graph shows the mean fixation values over time, excluding data points that
         exceed 2 standard deviations from the mean in each time interval.
        2. Standard Deviation Over Time: This graph displays the standard deviation of fixation values over time,
         also excluding data points that exceed 2 standard deviations from the mean.

    """
    (good_eyesight_stdev,
     good_eyesight_sem,
     good_eyesight_fixation_average_sorted_by_time) = get_group_data_for_graphs(
        good_analyzed_experiments
    )
    (bad_eyesight_stdev,
     bad_eyesight_sem,
     bad_eyesight_fixation_average_sorted_by_time) = get_group_data_for_graphs(
        bad_analyzed_experiments
    )

    with open(os.path.join("assets", "fixation_average_sorted_by_time.csv"), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(good_eyesight_fixation_average_sorted_by_time)
        writer.writerow(bad_eyesight_fixation_average_sorted_by_time)

    fig_fixation_count, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {
            str(Eyesight.GOOD): good_eyesight_fixation_average_sorted_by_time,
            str(Eyesight.BAD): bad_eyesight_fixation_average_sorted_by_time
        },
        {
            str(Eyesight.GOOD): GREEN,
            str(Eyesight.BAD): RED
        },
        "Time",
        "Fixations (Average)",
        "Average Number of Fixations\n(Excluding Data Beyond 2 Standard Deviations)",
        GraphType.Line,
        [str(Eyesight.GOOD), str(Eyesight.BAD)],
        add_fill_between=True,
        fill_between_values={
            str(Eyesight.GOOD): good_eyesight_sem,
            str(Eyesight.BAD): bad_eyesight_sem
        }
    )

    fig_fixation_count_stdev, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {
            "Good Eyesight Standard deviation": good_eyesight_stdev,
            "Bad Eyesight Standard deviation": bad_eyesight_stdev
        },
        {
            "Good Eyesight Standard deviation": DARK_GREEN,
            "Bad Eyesight Standard deviation": DARK_RED
        },
        "Time",
        "Fixations Count Standard Deviation",
        "Fixations Count Standard Deviation\n(Excluding Data Beyond 2 Standard Deviations)",
        GraphType.Line,
        ["Good Eyesight Standard deviation", "Bad Eyesight Standard deviation"]
    )
    return fig_fixation_count, fig_fixation_count_stdev


def get_blink_graphs(good_analyzed_experiments: AnalyzedExperiments,
                     bad_analyzed_experiments: AnalyzedExperiments,
                     ids_order):
    def get_analyzed_experiment_data_for_blink_graphs(analyzed_experiments: AnalyzedExperiments):
        (num_of_blink_mean,
         blink_num_mean,
         blinks_sem,
         duration_sem) = analyzed_experiments.get_mean_number_of_blinks_and_duration()

        analyzed_experiments_single_experiments_num_blinks: dict[str: int] = {
            analyzed_experiment.experiment.experiment_real_id:
                analyzed_experiment.get_num_of_blinks_and_mean_duration_in_video()[0]
            for analyzed_experiment
            in sorted(analyzed_experiments.analyzed_experiments.values(),
                      key=lambda item: item.experiment.experiment_real_id)
        }
        return (num_of_blink_mean, blink_num_mean, analyzed_experiments_single_experiments_num_blinks,
                blinks_sem, duration_sem)

    (good_num_of_blink_mean,
     good_blink_num_mean,
     good_experiments_single_experiments_num_blinks,
     good_blinks_sem,
     good_duration_sem) = get_analyzed_experiment_data_for_blink_graphs(
        good_analyzed_experiments
    )
    (bad_num_of_blink_mean,
     bad_blink_num_mean,
     bad_experiments_single_experiments_num_blinks,
     bad_blinks_sem,
     bad_duration_sem) = get_analyzed_experiment_data_for_blink_graphs(
        bad_analyzed_experiments
    )

    t_stat_num_blinks, p_value_num_blinks = stats.ttest_rel(
        [bad_experiments_single_experiments_num_blinks[id_] for id_ in ids_order],
        [good_experiments_single_experiments_num_blinks[id_] for id_ in ids_order]
    )

    correlation_num_blinks = np.corrcoef(
        [bad_experiments_single_experiments_num_blinks[id_] for id_ in ids_order],
        [good_experiments_single_experiments_num_blinks[id_] for id_ in ids_order]
    )[0, 1]

    num_blink_differences: dict[int: int] = {
        experiment_real_id: good_num_duration - bad_experiments_single_experiments_num_blinks[experiment_real_id]
        for experiment_real_id, good_num_duration in good_experiments_single_experiments_num_blinks.items()
    }

    mean_num_of_blinks_fig, _ = create_bar_graph(
        [str(Eyesight.GOOD), str(Eyesight.BAD)],
        [good_num_of_blink_mean, bad_num_of_blink_mean],
        "Eyesight",
        "Blinks (Average)",
        f"Average Number of Blinks\nt-statistic: {t_stat_num_blinks:.3f},"
        f" p-value: {p_value_num_blinks:.3f},",
        # f" correlation: {correlation_num_blinks:.3f}",
        y_errors=[good_blinks_sem, bad_blinks_sem]
    )

    mean_duration_fig, _ = create_bar_graph(
        [str(Eyesight.GOOD), str(Eyesight.BAD)],
        [good_blink_num_mean, bad_blink_num_mean],
        "Eyesight",
        "Blink Duration (Average)",
        "Average Blink Duration [ms]",
        y_errors=[good_duration_sem, bad_duration_sem]
    )

    single_experiments_num_of_blinks_fig, _ = create_scattered_graph(
        {
            str(Eyesight.GOOD): [
                ScreenLocation(int(experiment_real_id), number_of_blinks)
                for experiment_real_id, number_of_blinks in good_experiments_single_experiments_num_blinks.items()
            ],
            str(Eyesight.BAD): [
                ScreenLocation(int(experiment_real_id), number_of_blinks)
                for experiment_real_id, number_of_blinks in bad_experiments_single_experiments_num_blinks.items()
            ]
        },
        {
            str(Eyesight.GOOD): GREEN,
            str(Eyesight.BAD): RED
        },
        "Participant",
        "Number of Blinks",
        "Number of Blinks per Participant",
    )

    num_of_blinks_differences, _ = create_scattered_graph(
        {
            "Differences per experiment": [
                ScreenLocation(int(experiment_real_id), num_blinks_differences)
                for experiment_real_id, num_blinks_differences in num_blink_differences.items()
            ]
        },
        {
            "Differences per experiment": LIGHT_BLUE
        },
        "Participant",
        "Number of Blinks Difference",
        "Number of Blinks Difference per Participant",
    )

    num_of_blinks_fig_y_good_x_bad = [
        ScreenLocation(
            bad_experiments_single_experiments_num_blinks[experiment_real_id],
            good_experiments_single_experiments_num_blinks[experiment_real_id]
        )
        for experiment_real_id in ids_order
    ]
    single_experiments_num_of_blinks_fig_y_good_x_bad, _ = create_scattered_graph(
        {
            "Experiment": num_of_blinks_fig_y_good_x_bad
        },
        {
            "Experiment": LIGHT_PURPLE
        },
        "Bad Eyesight Number of Blinks",
        "Good Eyesight Number of Blinks",
        f"Number of Blinks per Experiment"
        # f"\nt-statistic: {t_stat_num_blinks:.3f},"
        # f" p-value: {p_value_num_blinks:.3f},"
        f" correlation: {correlation_num_blinks:.3f}",
        add_middle_line=True,
        add_x_equal_y_line=True
    )

    return (mean_num_of_blinks_fig,
            mean_duration_fig,
            single_experiments_num_of_blinks_fig,
            num_of_blinks_differences,
            single_experiments_num_of_blinks_fig_y_good_x_bad,
            num_of_blinks_fig_y_good_x_bad)


def get_x_y_coordinates_through_time_graphs(good_analyzed_experiments: AnalyzedExperiments,
                                            bad_analyzed_experiments: AnalyzedExperiments):
    def create_gaze_y_or_x_graph_data(analyzed_experiments_group: AnalyzedExperiments, color: str):
        experiments_x = {}
        experiments_y = {}
        for experiment_id, analyzed_experiment in analyzed_experiments_group.analyzed_experiments.items():
            indexes_x = [screen_location.x for screen_location in analyzed_experiment.average_screen_locations]
            indexes_y = [screen_location.y for screen_location in analyzed_experiment.average_screen_locations]
            experiments_x[experiment_id] = indexes_x
            experiments_y[experiment_id] = indexes_y

        experiment_average_x: list[float] = [
            index.x for index in analyzed_experiments_group.average_screen_locations_sorted_by_time(MAX_DEVIATION)
        ]

        experiment_average_y: list[float] = [
            index.y for index in analyzed_experiments_group.average_screen_locations_sorted_by_time(MAX_DEVIATION)
        ]

        (stdev_gaze_locations_sorted_by_time_x,
         sem_gaze_locations_sorted_by_time_x,
         stdev_gaze_locations_sorted_by_time_y,
         sem_gaze_locations_sorted_by_time_y) = (
            analyzed_experiments_group.sem_gaze_locations_sorted_by_time(MAX_DEVIATION)
        )

        experiments_id_to_color = {
            experiment_id: color for experiment_id
            in analyzed_experiments_group.experiment_gaze_variances_sorted_by_time.keys()
        }

        return (experiments_x,
                experiments_y,
                experiment_average_x,
                experiment_average_y,
                experiments_id_to_color,
                stdev_gaze_locations_sorted_by_time_x,
                sem_gaze_locations_sorted_by_time_x,
                stdev_gaze_locations_sorted_by_time_y,
                sem_gaze_locations_sorted_by_time_y)

    def create_fixations_y_or_x_graph_data(analyzed_experiments_group: AnalyzedExperiments, color: str):
        experiments_x = {}
        experiments_y = {}
        for experiment_id, analyzed_experiment in analyzed_experiments_group.analyzed_experiments.items():
            indexes_x = [
                screen_location.x if screen_location is not None else None
                for screen_location in analyzed_experiment.average_fixation_locations
            ]
            indexes_y = [
                screen_location.y if screen_location is not None else None
                for screen_location in analyzed_experiment.average_fixation_locations
            ]
            experiments_x[experiment_id] = indexes_x
            experiments_y[experiment_id] = indexes_y

        experiment_average_x: list[float] = [
            index.x if index is not None else None
            for index in analyzed_experiments_group.average_fixation_locations_sorted_by_time(MAX_DEVIATION)
        ]

        (sem_fixation_locations_sorted_by_time_x,
         sem_fixation_locations_sorted_by_time_y) = (
            analyzed_experiments_group.sem_fixation_locations_sorted_by_time(MAX_DEVIATION)
        )

        experiment_average_y: list[float] = [
            index.y if index is not None else None
            for index in analyzed_experiments_group.average_fixation_locations_sorted_by_time(MAX_DEVIATION)
        ]

        experiments_id_to_color = {
            experiment_id: color for experiment_id
            in analyzed_experiments_group.experiment_fixation_variances_sorted_by_time.keys()
        }

        return (experiments_x, experiments_y, experiment_average_x, experiment_average_y, experiments_id_to_color,
                sem_fixation_locations_sorted_by_time_x, sem_fixation_locations_sorted_by_time_y)

    """
    :param good_analyzed_experiments: AnalyzedExperiments class of experiment of people with bad eyesight
    :param bad_analyzed_experiments: AnalyzedExperiments class of experiment of people with good eyesight.
    :return: Graphs displaying the x-axis and y-axis locations over time. The average is highlighted in a different
     color, and for each timestamp (in the average only), the data points included are those that do not exceed
      the specified number of standard deviations from the mean.
    """

    (good_experiments_x_gaze,
     good_experiments_y_gaze,
     good_experiment_average_x_gaze,
     good_experiment_average_y_gaze,
     good_experiments_id_to_color_gaze,
     good_gaze_x_stdev_sorted_by_time,
     good_gaze_x_sem_sorted_by_time,
     good_gaze_y_stdev_sorted_by_time,
     good_gaze_y_sem_sorted_by_time) = create_gaze_y_or_x_graph_data(good_analyzed_experiments, LIGHT_GREY)
    (bad_experiments_x_gaze,
     bad_experiments_y_gaze,
     bad_experiment_average_x_gaze,
     bad_experiment_average_y_gaze,
     bad_experiments_id_to_color_gaze,
     bad_gaze_x_stdev_sorted_by_time,
     bad_gaze_x_sem_sorted_by_time,
     bad_gaze_y_stdev_sorted_by_time,
     bad_gaze_y_sem_sorted_by_time) = create_gaze_y_or_x_graph_data(bad_analyzed_experiments, LIGHT_GREY)

    x_values_gaze_fix, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {
            **good_experiments_x_gaze,
            **bad_experiments_x_gaze,
            "Good Eyesight Average": good_experiment_average_x_gaze,
            "Bad Eyesight Average": bad_experiment_average_x_gaze
        },
        {
            **good_experiments_id_to_color_gaze,
            **bad_experiments_id_to_color_gaze,
            "Good Eyesight Average": GREEN,
            "Bad Eyesight Average": RED
        },
        "Time",
        "X Value",
        "X-Axis Values (Gaze)\n(Excluding Data Beyond 2 Standard Deviations)",
        GraphType.Line,
        ["Good Eyesight Average", "Bad Eyesight Average"],
        add_fill_between=True,
        fill_between_values={
            "Good Eyesight Average": good_gaze_x_sem_sorted_by_time,
            "Bad Eyesight Average": bad_gaze_x_sem_sorted_by_time
        }
    )

    y_values_gaze_fig, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {
            **good_experiments_y_gaze,
            **bad_experiments_y_gaze,
            "Good Eyesight Average": good_experiment_average_y_gaze,
            "Bad Eyesight Average": bad_experiment_average_y_gaze
        },
        {
            **good_experiments_id_to_color_gaze,
            **bad_experiments_id_to_color_gaze,
            "Good Eyesight Average": GREEN,
            "Bad Eyesight Average": RED
        },
        "Time",
        "Y Value",
        "Y-Axis Values (Gaze)\n(Excluding Data Beyond 2 Standard Deviations)",
        GraphType.Line,
        ["Good Eyesight Average", "Bad Eyesight Average"],
        add_fill_between=True,
        fill_between_values={
            "Good Eyesight Average": good_gaze_y_sem_sorted_by_time,
            "Bad Eyesight Average": bad_gaze_y_sem_sorted_by_time
        }
    )

    (good_experiments_x_fixation,
     good_experiments_y_fixation,
     good_experiment_average_x_fixation,
     good_experiment_average_y_fixation,
     good_experiments_id_to_color_fixation,
     good_fixation_x_sem_sorted_by_time,
     good_fixation_y_sem_sorted_by_time) = create_fixations_y_or_x_graph_data(good_analyzed_experiments, LIGHT_GREY)
    (bad_experiments_x_fixation,
     bad_experiments_y_fixation,
     bad_experiment_average_x_fixation,
     bad_experiment_average_y_fixation,
     bad_experiments_id_to_color_fixation,
     bad_fixation_x_sem_sorted_by_time,
     bad_fixation_y_sem_sorted_by_time
     ) = create_fixations_y_or_x_graph_data(bad_analyzed_experiments, LIGHT_GREY)

    x_values_fixations_fig, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {
            **good_experiments_x_fixation,
            **bad_experiments_x_fixation,
            "Good Eyesight Average": good_experiment_average_x_fixation,
            "Bad Eyesight Average": bad_experiment_average_x_fixation
        },
        {
            **good_experiments_id_to_color_fixation,
            **bad_experiments_id_to_color_fixation,
            "Good Eyesight Average": GREEN,
            "Bad Eyesight Average": RED
        },
        "Time",
        "X Value",
        "X-Axis Values (Fixations)\n(Excluding Data Beyond 2 Standard Deviations)",
        GraphType.Line,
        ["Good Eyesight Average", "Bad Eyesight Average"],
        add_fill_between=True,
        fill_between_values={
            "Good Eyesight Average": good_fixation_x_sem_sorted_by_time,
            "Bad Eyesight Average": bad_fixation_x_sem_sorted_by_time
        }
    )

    y_values_fixations_fig, _ = create_time_series_scattered_or_line_graph_sorted_by_time(
        {
            **good_experiments_y_fixation,
            **bad_experiments_y_fixation,
            "Good Eyesight Average": good_experiment_average_y_fixation,
            "Bad Eyesight Average": bad_experiment_average_y_fixation
        },
        {
            **good_experiments_id_to_color_fixation,
            **bad_experiments_id_to_color_fixation,
            "Good Eyesight Average": GREEN,
            "Bad Eyesight Average": RED
        },
        "Time",
        "y value",
        "Y-Axis Values (Fixations)\n(Excluding Data Beyond 2 Standard Deviations)",
        GraphType.Line,
        ["Good Eyesight Average", "Bad Eyesight Average"],
        add_fill_between=True,
        fill_between_values={
            "Good Eyesight Average": good_fixation_y_sem_sorted_by_time,
            "Bad Eyesight Average": bad_fixation_y_sem_sorted_by_time
        }
    )

    return x_values_gaze_fix, y_values_gaze_fig, x_values_fixations_fig, y_values_fixations_fig


# def plot_fft(good_analyzed_experiments: AnalyzedExperiments,
#              bad_analyzed_experiments: AnalyzedExperiments):
#     def run_fft(analyzed_experiments, values_col):
#         fft_values = []
#         fft_frequencies = []
#
#         group_gaze_points = [
#             analyzed_experiment.experiment.mapped_gaze_on_video.gaze[values_col]
#             for experiment_id, analyzed_experiment in analyzed_experiments.analyzed_experiments.items()
#         ]
#
#         min_length = min(len(x) for x in group_gaze_points)
#         group_gaze_points_short_length = [x[:min_length] for x in group_gaze_points]
#
#         all_gaze_points = np.concatenate(group_gaze_points_short_length)
#         global_mean = np.mean(all_gaze_points)
#
#         # Center data using global mean
#         centered_gaze_points = [x - global_mean for x in group_gaze_points_short_length]
#
#         mean_gaze_points = np.mean(np.array(centered_gaze_points), axis=0)
#         mean_gaze_points -= np.mean(mean_gaze_points)
#
#         for gaze_points in centered_gaze_points:
#             # Apply FFT
#             fft_values.append(np.fft.fft(gaze_points))
#
#             # Calculate the frequencies for plotting
#             fft_frequencies.append(np.fft.fftfreq(len(gaze_points)))
#
#         # Compute the mean FFT values for plotting
#         fft_frequencies.append(np.fft.fftfreq(min_length))
#         mean_fft_values = np.mean(np.abs(np.array(fft_values)), axis=0)
#         mean_fft_frequencies = fft_frequencies[0]
#
#         return mean_gaze_points, mean_fft_frequencies, mean_fft_values
#
#     def create_fft_graphs(mean_gaze_points, mean_fft_frequencies, mean_fft_values, group_name):
#         # Create figures for plotting
#         original_data_fig, original_data_ax = plt.subplots(figsize=(20, 6))
#         original_data_ax.plot(mean_gaze_points, marker='o', linestyle='-', color='b')
#         original_data_ax.set_title(f'Original Data {group_name}')
#         original_data_ax.set_xlabel('Index')
#         original_data_ax.set_ylabel('Value')
#         plt.tight_layout()
#
#         fft_mag_fig, fft_mag_ax = plt.subplots(figsize=(20, 6))
#         fft_mag_ax.stem(mean_fft_frequencies, mean_fft_values)
#         fft_mag_ax.set_title(f'FFT Magnitude {group_name}')
#         fft_mag_ax.set_xlabel('Frequency')
#         fft_mag_ax.set_ylabel('Magnitude')
#         plt.tight_layout()
#
#         # Return the figures and axes
#         return original_data_fig, original_data_ax, fft_mag_fig, fft_mag_ax
#
#     x_col = "gaze position transf x [px]"
#     y_col = "gaze position transf y [px]"
#
#     (good_x_mean_gaze_points,
#      good_x_mean_fft_frequencies,
#      good_x_mean_fft_values) = run_fft(good_analyzed_experiments, x_col)
#
#     (good_x_original_data_fig,
#      good_x_original_data_ax,
#      good_x_fft_mag_fig,
#      good_x_fft_mag_ax) = create_fft_graphs(
#         good_x_mean_gaze_points, good_x_mean_fft_frequencies, good_x_mean_fft_values, "Good Eyesight X-Axis"
#     )
#
#     (bad_x_mean_gaze_points,
#      bad_x_mean_fft_frequencies,
#      bad_x_mean_fft_values) = run_fft(bad_analyzed_experiments, x_col)
#
#     (bad_x_original_data_fig,
#      bad_x_original_data_ax,
#      bad_x_fft_mag_fig,
#      bad_x_fft_mag_ax) = create_fft_graphs(
#         bad_x_mean_gaze_points, bad_x_mean_fft_frequencies, bad_x_mean_fft_values, "Bad Eyesight X-Axis"
#     )
#
#     (good_y_mean_gaze_points,
#      good_y_mean_fft_frequencies,
#      good_y_mean_fft_values) = run_fft(good_analyzed_experiments, y_col)
#
#     (good_y_original_data_fig,
#      good_y_original_data_ax,
#      good_y_fft_mag_fig,
#      good_y_fft_mag_ax) = create_fft_graphs(
#         good_y_mean_gaze_points, good_y_mean_fft_frequencies, good_y_mean_fft_values, "Good Eyesight Y-Axis"
#     )
#
#     (bad_y_mean_gaze_points,
#      bad_y_mean_fft_frequencies,
#      bad_y_mean_fft_values) = run_fft(bad_analyzed_experiments, y_col)
#
#     (bad_y_original_data_fig,
#      bad_y_original_data_ax,
#      bad_y_fft_mag_fig,
#      bad_y_fft_mag_ax) = create_fft_graphs(
#         bad_y_mean_gaze_points, bad_y_mean_fft_frequencies, bad_y_mean_fft_values, "Bad Eyesight Y-Axis"
#     )
#
#     min_good_bad = min(len(good_y_mean_fft_values), len(bad_y_mean_fft_values))
#     bad_good_magnitude_difference_y = good_y_mean_fft_values[:min_good_bad] - bad_y_mean_fft_values[:min_good_bad]
#     bad_good_magnitude_difference_x = good_x_mean_fft_values[:min_good_bad] - bad_x_mean_fft_values[:min_good_bad]
#
#     fft_dif_mag_fig_x, fft_dif_mag_ax_x = plt.subplots(figsize=(20, 6))
#     fft_dif_mag_ax_x.stem(good_x_mean_fft_frequencies[:min_good_bad], bad_good_magnitude_difference_x)
#     fft_dif_mag_ax_x.set_title(f'FFT Magnitude Good - Bad Differences X-Axis')
#     fft_dif_mag_ax_x.set_xlabel('Frequency')
#     fft_dif_mag_ax_x.set_ylabel('Magnitude')
#     plt.tight_layout()
#
#     fft_dif_mag_fig_y, fft_dif_mag_ax_y = plt.subplots(figsize=(20, 6))
#     fft_dif_mag_ax_y.stem(good_y_mean_fft_frequencies[:min_good_bad], bad_good_magnitude_difference_y)
#     fft_dif_mag_ax_y.set_title(f'FFT Magnitude Good - Bad Differences Y-Axis')
#     fft_dif_mag_ax_y.set_xlabel('Frequency')
#     fft_dif_mag_ax_y.set_ylabel('Magnitude')
#     plt.tight_layout()
#
#     return (good_x_original_data_fig,
#             good_x_fft_mag_fig,
#             bad_x_original_data_fig,
#             bad_x_fft_mag_fig,
#             good_y_original_data_fig,
#             good_y_fft_mag_fig,
#             bad_y_original_data_fig,
#             bad_y_fft_mag_fig,
#             fft_dif_mag_fig_x,
#             fft_dif_mag_fig_y)

def plot_fft(good_analyzed_experiments: AnalyzedExperiments,
             bad_analyzed_experiments: AnalyzedExperiments):
    def run_fft(analyzed_experiments, values_col):
        fft_values = []
        fft_frequencies = []

        group_gaze_points = [
            analyzed_experiment.experiment.mapped_gaze_on_video.gaze[values_col]
            for experiment_id, analyzed_experiment in analyzed_experiments.analyzed_experiments.items()
        ]

        min_length = min(len(x) for x in group_gaze_points)
        group_gaze_points_short_length = [x[:min_length] for x in group_gaze_points]

        # todo maybe fix?
        all_gaze_points = np.concatenate(group_gaze_points_short_length)
        mean_gaze_points = np.mean(np.array(all_gaze_points), axis=0)

        for gaze_points in group_gaze_points_short_length:
            gaze_points = np.asarray(gaze_points, dtype=np.float64, order='C')
            yf = fft(gaze_points)
            xf = fftfreq(len(gaze_points), 1 / SAMPLE_RATE)

            fft_values.append(yf)
            fft_frequencies.append(xf)

            temp_fig, temp_ax = plt.subplots(figsize=(20, 6))
            temp_ax.plot(xf, np.abs(yf))
            temp_ax.set_title(f'FFT Magnitude try')
            temp_ax.set_xlabel('Frequency')
            temp_ax.set_ylabel('Magnitude')
            plt.tight_layout()
            temp_fig.savefig("1.pdf")

        mean_fft_values = np.mean(np.abs(fft_values), axis=0)

        fft_mag_fig, fft_mag_ax = plt.subplots(figsize=(20, 6))
        fft_mag_ax.plot(xf, mean_fft_values)
        fft_mag_ax.set_title('Mean FFT Magnitude')
        fft_mag_ax.set_xlabel('Frequency')
        fft_mag_ax.set_ylabel('Magnitude')
        plt.tight_layout()

        return mean_gaze_points, mean_fft_values, fft_mag_fig, xf

    x_col = "gaze position transf x [px]"
    y_col = "gaze position transf y [px]"

    (good_x_mean_gaze_points,
     good_x_mean_fft_values,
     good_x_fft_mag_fig,
     good_x_fft_frequencies) = run_fft(good_analyzed_experiments, x_col)

    (bad_x_mean_gaze_points,
     bad_x_mean_fft_values,
     bad_x_fft_mag_fig,
     bad_x_fft_frequencies) = run_fft(bad_analyzed_experiments, x_col)

    (good_y_mean_gaze_points,
     good_y_mean_fft_values,
     good_y_fft_mag_fig,
     good_y_fft_frequencies) = run_fft(good_analyzed_experiments, y_col)

    (bad_y_mean_gaze_points,
     bad_y_mean_fft_values,
     bad_y_fft_mag_fig,
     bad_y_fft_frequencies) = run_fft(bad_analyzed_experiments, y_col)

    min_good_bad = min(len(good_y_mean_fft_values), len(bad_y_mean_fft_values))
    bad_good_magnitude_difference_y = good_y_mean_fft_values[:min_good_bad] - bad_y_mean_fft_values[:min_good_bad]
    bad_good_magnitude_difference_x = good_x_mean_fft_values[:min_good_bad] - bad_x_mean_fft_values[:min_good_bad]

    fft_dif_mag_fig_x, fft_dif_mag_ax_x = plt.subplots(figsize=(20, 6))
    fft_dif_mag_ax_x.plot(good_x_fft_frequencies[:min_good_bad], bad_good_magnitude_difference_x)
    fft_dif_mag_ax_x.set_title('FFT Magnitude Good - Bad Differences X-Axis')
    fft_dif_mag_ax_x.set_xlabel('Frequency')
    fft_dif_mag_ax_x.set_ylabel('Magnitude')
    plt.tight_layout()

    fft_dif_mag_fig_y, fft_dif_mag_ax_y = plt.subplots(figsize=(20, 6))
    fft_dif_mag_ax_y.plot(good_x_fft_frequencies[:min_good_bad], bad_good_magnitude_difference_y)
    fft_dif_mag_ax_y.set_title('FFT Magnitude Good - Bad Differences Y-Axis')
    fft_dif_mag_ax_y.set_xlabel('Frequency')
    fft_dif_mag_ax_y.set_ylabel('Magnitude')
    plt.tight_layout()

    return (good_x_fft_mag_fig,
            bad_x_fft_mag_fig,
            good_y_fft_mag_fig,
            bad_y_fft_mag_fig,
            fft_dif_mag_fig_x,
            fft_dif_mag_fig_y)
