from enum import Enum
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from AnalyzedExperiments import AnalyzedExperiments
from itertools import zip_longest

RED: str = "#FF6161"
GREEN: str = "#68D47A"
BLUE: str = "#6C79CB"


class GraphType(Enum):
    Scattered = 1
    Line = 2


def return_bar_graph(categories: list[str],
                     values: list[float | int],
                     x_label: str,
                     y_label: str,
                     graph_title: str, bar_width=0.4) -> tuple[plt.figure, any]:
    fig, ax = plt.subplots()
    ax.bar(categories, values, width=bar_width, align='center', color=BLUE)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(graph_title)
    return fig, ax


def return_scattered_or_line_graph(group_name_to_locations: dict[str, list[float]],
                                   group_name_to_color: dict[str, str],
                                   x_label: str,
                                   y_label: str,
                                   title: str,
                                   graph_type: GraphType) -> tuple[plt.figure, any]:
    """
    :param group_name_to_locations: group name to it's values sorted by time, if for a certain time there's no data
                                fill it with None - all groups must have the same list length.
                                The same group can have several lists.
    :param group_name_to_color: A dictionary from each group name to it's color (RGB or known color names).
    :param y_label: The Y axis title.
    :param x_label: The X axis title.
    :param title: The graph title.
    :param graph_type:
    :return: The new figure and the ax_dict (for more info look at matplotlib.subplots documentation)
    """
    fig, ax = plt.subplots()
    for group_name, locations in group_name_to_locations.items():
        # Create a scatter plot for each group
        if graph_type == GraphType.Scattered:
            ax.scatter([x for x in range(1, len(locations) + 1)],
                       locations, color=group_name_to_color[group_name],
                       label=group_name, s=20)
        else:
            ax.plot([x for x in range(1, len(locations) + 1)],
                    locations, color=group_name_to_color[group_name],
                    label=group_name)

        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    # Add a legend
    ax.legend()

    return fig, ax


def matplotlib_figures_to_pdf(pdf_name: str, *figures: plt.figure) -> None:
    """
    :param pdf_name: The name of the PDF file.
    :param figures: Figure of the graphs you want to present in the PDF file.
    :return: None, but creates a PDF with the given figures.
    """
    pdf_pages: PdfPages = PdfPages(f"{pdf_name}.pdf")
    for fig in figures:
        pdf_pages.savefig(fig)
    pdf_pages.close()


def create_graphs_of_good_vs_bad_eyesight_fixation_data(
        good_analyzed_experiments: AnalyzedExperiments,
        bad_analyzed_experiments: AnalyzedExperiments) -> tuple[plt.figure, plt.figure]:
    """
    :param bad_analyzed_experiments: AnalyzedExperiments class of experiment of people with good eyesight.
    :param good_analyzed_experiments: AnalyzedExperiments class of experiment of people with bad eyesight
    :return: A graph representing the mean number of fixations and the mean duration of it, divided by
             good and bad eyesight.
    """
    (good_num_of_fixation_mean,
     good_duration_mean) = good_analyzed_experiments.get_mean_number_of_fixations_and_duration()

    (bad_num_of_fixation_mean,
     bad_duration_mean) = bad_analyzed_experiments.get_mean_number_of_fixations_and_duration()

    fixations_fig, _ = return_bar_graph(
        ["Good Eyesight", "Bad Eyesight"],
        [good_num_of_fixation_mean, bad_num_of_fixation_mean],
        "Eyesight",
        "Number of Fixations (Average)",
        "Average Number of Fixations"
    )

    duration_fig, _ = return_bar_graph(
        ["Good Eyesight", "Bad Eyesight"],
        [good_duration_mean, bad_duration_mean],
        "Eyesight",
        "Average Duration",
        "Average Fixation Duration [ms]"
    )

    return fixations_fig, duration_fig


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

    fig_variance, ax = return_scattered_or_line_graph(
        {**good_analyzed_experiments.experiment_gaze_variances_sorted_by_time,
         **bad_analyzed_experiments.experiment_gaze_variances_sorted_by_time},
        {**good_experiments_id_to_color, **bad_experiments_id_to_color},
        "Time",
        "Gaze Variance",
        "Variance (Screen Coordinates)",
        GraphType.Scattered
    )

    # create scatter plot of all of the experiment's mean variance.
    good_variances_means: list[float | None] = calculate_mean_of_lists_values(
        [variances for variances
         in good_analyzed_experiments.experiment_gaze_variances_sorted_by_time.values()]
    )
    bad_variances_means: list[float | None] = calculate_mean_of_lists_values(
        [variances for variances
         in bad_analyzed_experiments.experiment_gaze_variances_sorted_by_time.values()]
    )

    fig_variance_mean, _ = return_scattered_or_line_graph(
        {"Good Eyesight": good_variances_means, "Bad Eyesight": bad_variances_means},
        {"Good Eyesight": GREEN, "Bad Eyesight": RED},
        "Time passed",
        "Gaze Variance (Average)",
        "Average Gaze Variance (Screen Coordinates)",
        GraphType.Scattered
    )

    return fig_variance, fig_variance_mean


def get_fixations_variance_graphs(good_analyzed_experiments: AnalyzedExperiments,
                                  bad_analyzed_experiments: AnalyzedExperiments):
    good_eyesight_fixation_count_sorted_by_time = good_analyzed_experiments.fixation_count_sorted_by_time
    bad_eyesight_fixation_count_sorted_by_time = bad_analyzed_experiments.fixation_count_sorted_by_time

    fig_fixation_count, _ = return_scattered_or_line_graph(
        {"Good Eyesight": good_eyesight_fixation_count_sorted_by_time,
         "Bad Eyesight": bad_eyesight_fixation_count_sorted_by_time},
        {"Good Eyesight": GREEN, "Bad Eyesight": RED},
        "Time",
        "Fixations",
        "Number of Fixations",
        GraphType.Line
    )

    return fig_fixation_count


def get_blink_graphs(good_analyzed_experiments: AnalyzedExperiments,
                     bad_analyzed_experiments: AnalyzedExperiments):

    (good_num_of_blink_mean,
     good_blink_num_mean) = good_analyzed_experiments.get_mean_number_of_blinks_and_duration()

    (bad_num_of_blink_mean,
     bad_blink_num_mean) = bad_analyzed_experiments.get_mean_number_of_blinks_and_duration()

    mean_num_of_blinks_fig, _ = return_bar_graph(["Good Eyesight", "Bad Eyesight"],
                                                 [good_num_of_blink_mean, bad_num_of_blink_mean],
                                                 "Eyesight",
                                                 "Blinks (Average)",
                                                 "Average Number of Blinks")

    mean_duration_fig, _ = return_bar_graph(["Good Eyesight", "Bad Eyesight"],
                                            [good_blink_num_mean, bad_blink_num_mean],
                                            "Eyesight",
                                            "Blink Duration (Average)",
                                            "Average Blink Duration [ms]")
    return mean_num_of_blinks_fig, mean_duration_fig
