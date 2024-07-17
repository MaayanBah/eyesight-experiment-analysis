from experiment_types import Experiment, ExperimentInput
from client.get_input import get_user_input
from client.dialog_methods import show_yes_no_dialog
from graphs import (create_graphs_of_good_vs_bad_eyesight_fixation_data,
                    matplotlib_figures_to_pdf, get_gaze_variance_graphs, get_blink_graphs,
                    get_fixations_number_graphs,
                    create_fixations_count_and_duration_k_means_graph,
                    get_x_y_coordinates_through_time_graphs)
from AnalyzedExperiments import (AnalyzedExperiments, AnalyzedExperimentsParameters,
                                 split_experiments_by_eyesight, get_mapped_gaze_start_time_to_end_time,
                                 get_raw_data_fixation_start_time_to_end_time)
import matplotlib as plt


def get_the_analyzed_experiments_data(
        parsed_experiments: list[Experiment]) -> tuple[AnalyzedExperiments, AnalyzedExperiments]:
    nanoseconds_in_second: int = 10 ** 9

    good_eyesight_experiments, bad_eyesight_experiments = split_experiments_by_eyesight(parsed_experiments)
    parameters: AnalyzedExperimentsParameters = AnalyzedExperimentsParameters(
        *get_mapped_gaze_start_time_to_end_time(parsed_experiments),
        *get_raw_data_fixation_start_time_to_end_time(parsed_experiments),
        delta_time=nanoseconds_in_second
    )

    good_analyzed_experiments: AnalyzedExperiments = AnalyzedExperiments(good_eyesight_experiments, parameters)
    bad_analyzed_experiments: AnalyzedExperiments = AnalyzedExperiments(bad_eyesight_experiments, parameters)

    return good_analyzed_experiments, bad_analyzed_experiments


def create_graphs(good_analyzed_experiments: AnalyzedExperiments,
                  bad_analyzed_experiments: AnalyzedExperiments) -> list[plt.figure]:
    (num_of_fixations_fig,
     fixation_duration_fig,
     single_experiments_num_fixations,
     fixation_differences_fig,
     single_experiments_num_fixations_y_good_x_bad,
     single_experiments_duration_mean,
     duration_mean_differences_fig,
     single_experiments_duration_mean_y_good_x_bad) = create_graphs_of_good_vs_bad_eyesight_fixation_data(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    (fixations_count_and_duration_divided_to_good_bad_graph,
     fixations_count_and_duration_k_means_graph) = create_fixations_count_and_duration_k_means_graph(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    num_of_fixations_fig_sorted_by_time, fig_fixation_count_stdev = get_fixations_number_graphs(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    variance_fig, variance_mean_fig = get_gaze_variance_graphs(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    (mean_num_of_blinks_fig,
     mean_blinks_duration_fig,
     single_experiments_num_of_blinks_fig,
     num_of_blinks_differences,
     single_experiments_num_of_blinks_fig_y_good_x_bad) = get_blink_graphs(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    (x_coordinates_through_time_gaze_fig,
     y_coordinates_through_time_gaze_fig,
     x_coordinates_through_time_fixations_fig,
     y_coordinates_through_time_fixations_fig,
     ) = get_x_y_coordinates_through_time_graphs(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    return [num_of_fixations_fig_sorted_by_time,
            fig_fixation_count_stdev,
            x_coordinates_through_time_fixations_fig,
            y_coordinates_through_time_fixations_fig,
            num_of_fixations_fig,
            fixation_duration_fig,
            single_experiments_num_fixations,
            fixation_differences_fig,
            single_experiments_num_fixations_y_good_x_bad,
            single_experiments_duration_mean,
            duration_mean_differences_fig,
            single_experiments_duration_mean_y_good_x_bad,
            fixations_count_and_duration_divided_to_good_bad_graph,
            fixations_count_and_duration_k_means_graph,
            mean_num_of_blinks_fig, mean_blinks_duration_fig,
            single_experiments_num_of_blinks_fig,
            num_of_blinks_differences,
            single_experiments_num_of_blinks_fig_y_good_x_bad,
            variance_fig,
            variance_mean_fig,
            x_coordinates_through_time_gaze_fig,
            y_coordinates_through_time_gaze_fig]


def main():
    # let the users choose their preferred input method and use it
    input_one_by_one: bool = show_yes_no_dialog("Do you want to insert the experiments one by one?")
    experiments_input: list[ExperimentInput] = get_user_input(one_by_one=input_one_by_one)

    parsed_experiments: list[Experiment] = [Experiment(experiment_input)
                                            for experiment_input in experiments_input]
    good_analyzed_experiments, bad_analyzed_experiments = get_the_analyzed_experiments_data(parsed_experiments)

    graphs: list[plt.figure] = create_graphs(good_analyzed_experiments,
                                             bad_analyzed_experiments)
    matplotlib_figures_to_pdf("assets/graphs",
                              *graphs)


if __name__ == "__main__":
    main()
