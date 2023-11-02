from experiment_types import Experiment, ExperimentInput
from client.get_input import get_user_input
from client.parse_input import parse_single_experiment_input
from client.dialog_methods import show_yes_no_dialog


def main():
    input_one_by_one: bool = show_yes_no_dialog("Do you want to insert the experiments one by one?")
    experiments_input: list[ExperimentInput] = get_user_input(one_by_one=input_one_by_one)
    parsed_experiments: list[Experiment] = [parse_single_experiment_input(experiment_input)
                                            for experiment_input in experiments_input]


if __name__ == "__main__":
    main()
