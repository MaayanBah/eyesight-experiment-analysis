from experiment_types import Experiment, ExperimentInput
from client.get_input import get_user_input
from client.dialog_methods import show_yes_no_dialog
from graphs import (create_graphs_of_good_vs_bad_eyesight_fixation_data,
                    matplotlib_figures_to_pdf, get_gaze_variance_graphs, get_blink_graphs,
                    get_fixations_variance_graphs,
                    create_fixations_count_and_duration_k_means_graph)
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
     fixation_duration_fig) = create_graphs_of_good_vs_bad_eyesight_fixation_data(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    (fixations_count_and_duration_divided_to_good_bad_graph,
     fixations_count_and_duration_k_means_graph) = create_fixations_count_and_duration_k_means_graph(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    num_of_fixations_fig_sorted_by_time = get_fixations_variance_graphs(good_analyzed_experiments,
                                                                        bad_analyzed_experiments)

    variance_fig, variance_mean_fig = get_gaze_variance_graphs(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    num_of_blinks_fig, blinks_duration_fig = get_blink_graphs(
        good_analyzed_experiments, bad_analyzed_experiments
    )

    return [num_of_fixations_fig_sorted_by_time,
            num_of_fixations_fig, fixation_duration_fig,
            fixations_count_and_duration_divided_to_good_bad_graph,
            fixations_count_and_duration_k_means_graph,
            num_of_blinks_fig, blinks_duration_fig,
            variance_fig, variance_mean_fig]


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
    # todo change df names to const names
    # todo parse input including dtypes and data modification


if __name__ == "__main__":
    main()
