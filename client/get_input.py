import os
from client.dialog_methods import display_message, get_directory_path, show_yes_no_dialog
from experiment_types import Eyesight, ExperimentInput

raw_data_must_have_files: set[str] = {"blinks.csv", "events.csv", "fixations.csv",
                                      "gaze.csv", "imu.csv", "world_timestamps.csv"}
reference_data_must_have_files: set[str] = {"fixations.csv", "gaze.csv", "sections.csv"}
mapped_gaze_must_have_files: set[str] = {"gaze.csv"}


def assert_necessary_files(directory_path: str,
                           must_have_files: set[str]) -> None:
    """
    :param directory_path: The directory's path.
    :param must_have_files: The files that the directory must contain.
                            Make sure to mention the files extensions.
    :return: None
    """
    directory_listdir: set[str] = set(os.listdir(directory_path))
    missing_files: set[str] = must_have_files.difference(directory_listdir)
    if len(missing_files) > 0:
        raise FileNotFoundError(f"The directory is missing the files: {missing_files}")


def get_single_experiment_directories() -> tuple[str, str, str]:
    display_message("Please choose the raw data directory")
    raw_data_directory_path: str = get_directory_path(title="Raw data directory")
    assert_necessary_files(raw_data_directory_path, raw_data_must_have_files)

    display_message("Please choose the reference data directory")
    reference_data_directory_path: str = get_directory_path(title="Reference data directory")
    assert_necessary_files(reference_data_directory_path, reference_data_must_have_files)

    display_message("Please choose the mapped gaze onto a display content directory")
    mapped_gaze_directory_path: str = get_directory_path(title="Mapped gaze directory")
    assert_necessary_files(mapped_gaze_directory_path, mapped_gaze_must_have_files)

    return raw_data_directory_path, reference_data_directory_path, mapped_gaze_directory_path


def get_one_by_one():
    experiments_inputs: list[ExperimentInput] = []

    still_getting_input: bool = True
    while still_getting_input:
        subject_eyesight: Eyesight = (Eyesight.GOOD
                                      if show_yes_no_dialog("Does the subject have good eyesight?")
                                      else
                                      Eyesight.BAD)
        experiments_inputs.append(ExperimentInput(subject_eyesight,
                                                  *get_single_experiment_directories()))
        still_getting_input = show_yes_no_dialog("Do you want to insert another experiment?")
    return experiments_inputs


def get_all_together():
    experiments_inputs: list[ExperimentInput] = []

    for subjects_eyesight in [Eyesight.GOOD, Eyesight.BAD]:
        display_message(f"Please choose the directory of experiments of subjects with {subjects_eyesight} eyesight")
        experiments_dir_path: str = get_directory_path(f"{subjects_eyesight} eyesight directory")

        for experiment in os.listdir(experiments_dir_path):
            experiment_dir: str = os.path.join(experiments_dir_path, experiment)

            raw_data_directory_path: str = os.path.join(experiment_dir, "raw_data")
            assert_necessary_files(raw_data_directory_path, raw_data_must_have_files)

            reference_data_directory_path: str = os.path.join(experiment_dir, "reference_data")
            assert_necessary_files(reference_data_directory_path, reference_data_must_have_files)

            mapped_gaze_directory_path: str = os.path.join(experiment_dir, "mapped_gaze")
            assert_necessary_files(mapped_gaze_directory_path, mapped_gaze_must_have_files)

            experiments_inputs.append(ExperimentInput(subjects_eyesight,
                                                      raw_data_directory_path,
                                                      reference_data_directory_path,
                                                      mapped_gaze_directory_path))
    return experiments_inputs


def get_user_input(one_by_one: bool) -> list[ExperimentInput]:
    """
    :param one_by_one: True if you want the user to choose each experiment in a different dialog box,
                       False if you want the user to choose a file of all the experiments together.
    :return: A list of the experiment input if it's valid, may raise exception.
    """
    return get_one_by_one() if one_by_one else get_all_together()
