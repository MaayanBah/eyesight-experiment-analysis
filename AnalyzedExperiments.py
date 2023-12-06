from dataclasses import dataclass
import pandas
from experiment_types import Experiment, Eyesight, ScreenLocation
from functools import lru_cache


def average_screen_location(screen_locations: set[ScreenLocation]) -> ScreenLocation | None:
    if len(screen_locations) == 0:
        return None
    sum_x: float = 0
    sum_y: float = 0
    for screen_location in screen_locations:
        sum_x += screen_location.x
        sum_y += screen_location.y
    return ScreenLocation(sum_x / len(screen_locations), sum_y / len(screen_locations))


def screen_location_variance(screen_locations: set[ScreenLocation],
                             average: ScreenLocation) -> float | None:
    if len(screen_locations) == 0:
        return None
    x_sigma: float = 0
    y_sigma: float = 0
    for screen_location in screen_locations:
        x_sigma += (screen_location.x - average.x) ** 2
        y_sigma += (screen_location.y - average.y) ** 2

    x_variance = x_sigma / len(screen_locations)
    y_variance = y_sigma / len(screen_locations)
    return x_variance + y_variance


def get_mapped_gaze_start_time_to_end_time(experiments: list[Experiment]) -> tuple[int, int]:
    latest_first_saccade_time_ns: int = max(
        experiment.mapped_gaze_on_video.gaze["time passed from start [ns]"].iloc[0]
        for experiment in experiments
    )

    earliest_last_saccade_time_ns: int = min(
        experiment.mapped_gaze_on_video.gaze["time passed from start [ns]"].iloc[-1]
        for experiment in experiments
    )

    return latest_first_saccade_time_ns, earliest_last_saccade_time_ns


def get_raw_data_fixation_start_time_to_end_time(experiments: list[Experiment]) -> tuple[int, int]:
    latest_first_fixation_time_ns: int = max(
        experiment.reference_data.fixations["start timestamp [ns]"].iloc[0]
        for experiment in experiments
    )

    earliest_last_fixation_time_ns: int = min(
        experiment.reference_data.fixations["start timestamp [ns]"].iloc[-1]
        for experiment in experiments
    )

    return latest_first_fixation_time_ns, earliest_last_fixation_time_ns


def split_experiments_by_eyesight(experiments: list[Experiment]) -> tuple[list[Experiment], list[Experiment]]:
    good_eyesight_experiments = []
    bad_eyesight_experiments = []

    for experiment in experiments:
        (good_eyesight_experiments
         if experiment.eyesight == Eyesight.GOOD
         else
         bad_eyesight_experiments).append(experiment)

    return good_eyesight_experiments, bad_eyesight_experiments


@dataclass
class AnalyzedExperimentsParameters:
    gaze_start_time: int
    gaze_end_time: int
    fixation_start_time: int
    fixation_end_time: int
    delta_time: int


def gaze_time_to_index(period_start_time: int,
                       parameters: AnalyzedExperimentsParameters) -> int:
    return (period_start_time - parameters.gaze_start_time) // parameters.delta_time


def fixation_time_to_index(period_start_time: int,
                           parameters: AnalyzedExperimentsParameters) -> int:
    return (period_start_time - parameters.fixation_start_time) // parameters.delta_time


class AnalyzedExperiment:
    def __init__(self,
                 experiment: Experiment,
                 parameters: AnalyzedExperimentsParameters):

        self.__experiment: Experiment = experiment

        self.__parameters: AnalyzedExperimentsParameters = parameters

        # self.__video_start_timestamp = self.experiment.raw_data.events.loc[
        #     self.experiment.raw_data.events["name"] == "start.video", "timestamp [ns]"
        # ].values[0]

        # self.__video_end_timestamp = self.experiment.raw_data.events.loc[
        #     self.experiment.raw_data.events["name"] == "end.video", "timestamp [ns]"
        # ].values[0]

    @property
    def experiment(self):
        return self.__experiment

    @property
    def parameters(self):
        return self.__parameters

    # @property
    # def video_start_timestamp(self):
    #     return self.__video_start_timestamp

    # @property
    # def video_end_timestamp(self):
    #     return self.__video_end_timestamp

    @lru_cache(maxsize=1)
    def get_num_of_fixation_and_mean_duration_in_video(self) -> tuple[int, int]:
        """
        :return: The number of fixations between the first and second events and the mean duration of these fixations
                 (in milliseconds).
        """
        # The fication df is already filtered within the specified time range of the video.
        fixation_df = self.experiment.reference_data.fixations

        # # Filter fixation data within the specified time range
        # fixations_during_event = fixation_df[
        #     (fixation_df["start timestamp [ns]"] >= self.video_start_timestamp) &
        #     (fixation_df["start timestamp [ns]"] <= self.video_end_timestamp) &
        #     (fixation_df["end timestamp [ns]"] >= self.video_start_timestamp) &
        #     (fixation_df["end timestamp [ns]"] <= self.video_end_timestamp)
        #      ]

        # Calculate the number of fixations and average fixation duration
        num_fixations = len(fixation_df)
        average_fixation_duration = fixation_df["duration [ms]"].mean()

        return num_fixations, average_fixation_duration

    @staticmethod
    def split_fixations_by_time(experiment: Experiment,
                                parameters: AnalyzedExperimentsParameters) -> list[set[ScreenLocation]]:
        fixations_locations_by_time: list[set[ScreenLocation]] = []

        for current_start_time in range(parameters.fixation_start_time,
                                        parameters.fixation_end_time,
                                        parameters.delta_time):
            experiment_fixation = experiment.reference_data.fixations
            fixations_in_time_period: pandas.DataFrame = experiment_fixation[
                (current_start_time <= experiment_fixation["start timestamp [ns]"])
                & (experiment_fixation["start timestamp [ns]"] < current_start_time + parameters.delta_time)
            ]

            locations_in_time_period = set(fixations_in_time_period.apply(
                lambda row: ScreenLocation(row["fixation x [px]"],
                                           row["fixation y [px]"]),
                axis=1)
            )

            fixations_locations_by_time.append(locations_in_time_period)
        return fixations_locations_by_time

    @lru_cache(maxsize=1)
    def get_num_of_blinks_and_mean_duration_in_video(self) -> tuple[int, int]:
        blinks_df = self.experiment.raw_data.blinks

        # Filter blinks data within the specified time range
        blinks_during_event = blinks_df[
            (blinks_df["start timestamp [ns]"] >= self.experiment.video_start_timestamp) &
            (blinks_df["start timestamp [ns]"] <= self.experiment.video_end_timestamp) &
            (blinks_df["end timestamp [ns]"] >= self.experiment.video_start_timestamp) &
            (blinks_df["end timestamp [ns]"] <= self.experiment.video_end_timestamp)
        ]

        # Calculate the number of blinks and average blinks duration
        num_blinks = len(blinks_during_event)
        average_blinks_duration = blinks_during_event["duration [ms]"].mean()

        return num_blinks, average_blinks_duration

    @staticmethod
    def split_screen_locations_by_time(experiment: Experiment,
                                       parameters: AnalyzedExperimentsParameters) -> list[set[ScreenLocation]]:
        screen_locations_by_time: list[set[ScreenLocation]] = []

        for current_start_time in range(parameters.gaze_start_time, parameters.gaze_end_time, parameters.delta_time):
            experiment_gaze = experiment.mapped_gaze_on_video.gaze
            gazes_in_time_period: pandas.DataFrame = experiment_gaze[
                (current_start_time <= experiment_gaze["time passed from start [ns]"])
                & (experiment_gaze["time passed from start [ns]"] < current_start_time + parameters.delta_time)
            ]

            locations_in_time_period = set(gazes_in_time_period.apply(
                lambda row: ScreenLocation(row["gaze position transf x [px]"],
                                           row["gaze position transf y [px]"]),
                axis=1)
            )

            screen_locations_by_time.append(locations_in_time_period)

        return screen_locations_by_time

    @property
    @lru_cache(maxsize=1)
    def screen_locations_sorted_by_time(self):
        screen_locations_sorted_by_time: list[set[ScreenLocation]] = (
            AnalyzedExperiment.split_screen_locations_by_time(self.experiment, self.parameters)
        )
        return screen_locations_sorted_by_time

    @property
    @lru_cache(maxsize=1)
    def fixation_locations_sorted_by_time(self):
        fixation_locations_sorted_by_time: list[set[ScreenLocation]] = (
            AnalyzedExperiment.split_fixations_by_time(self.experiment, self.parameters)
        )
        return fixation_locations_sorted_by_time

    @property
    @lru_cache(maxsize=1)
    def average_screen_locations(self) -> list[ScreenLocation]:
        average_screen_locations: list[ScreenLocation] = [
            average_screen_location(screen_locations)
            for screen_locations in self.screen_locations_sorted_by_time
        ]
        return average_screen_locations

    @property
    @lru_cache(maxsize=1)
    def average_fixation_locations(self) -> list[ScreenLocation]:
        average_fixation_locations: list[ScreenLocation] = [
            average_screen_location(screen_locations)
            for screen_locations in self.fixation_locations_sorted_by_time
        ]
        return average_fixation_locations


class AnalyzedExperiments:
    def __init__(self,
                 experiments: list[Experiment],
                 parameters: AnalyzedExperimentsParameters):
        self.__parameters = parameters
        self.__experiments = experiments

        self.__analyzed_experiments: dict[int, AnalyzedExperiment] = {
            experiment.id: AnalyzedExperiment(experiment, self.__parameters)
            for experiment in self.__experiments
        }

    @property
    def parameters(self):
        return self.__parameters

    @property
    def experiments(self):
        return self.__experiments

    @property
    def analyzed_experiments(self):
        return self.__analyzed_experiments

    @property
    @lru_cache(maxsize=1)
    def average_screen_locations_sorted_by_time(self):
        average_screen_locations_sorted_by_time: list[ScreenLocation] = [
            average_screen_location(
                {
                    screen_location
                    for analyzed_experiment in self.__analyzed_experiments.values()
                    for screen_location in analyzed_experiment.screen_locations_sorted_by_time[
                     gaze_time_to_index(period_start_time, self.__parameters)
                     ]
                }
            )
            for period_start_time in range(self.__parameters.gaze_start_time,
                                           self.__parameters.gaze_end_time,
                                           self.__parameters.delta_time)
        ]

        return average_screen_locations_sorted_by_time

    @property
    @lru_cache(maxsize=1)
    def average_fixation_locations_sorted_by_time(self):
        average_fixation_locations_sorted_by_time: list[ScreenLocation] = [
            average_screen_location(
                {
                    screen_location
                    for analyzed_experiment in self.__analyzed_experiments.values()
                    for screen_location in analyzed_experiment.fixation_locations_sorted_by_time[
                     fixation_time_to_index(period_start_time, self.__parameters)
                     ]
                }
            )
            for period_start_time in range(self.__parameters.fixation_start_time,
                                           self.__parameters.fixation_end_time,
                                           self.__parameters.delta_time)
        ]

        return average_fixation_locations_sorted_by_time

    @property
    @lru_cache(maxsize=1)
    def fixation_count_sorted_by_time(self) -> list[int]:
        fixation_count_sorted_by_time: list[int] = [
            sum(
                    len(analyzed_experiment.fixation_locations_sorted_by_time[
                     fixation_time_to_index(period_start_time, self.__parameters)
                        ])
                    for analyzed_experiment in self.__analyzed_experiments.values()
            )
            for period_start_time in range(self.__parameters.fixation_start_time,
                                           self.__parameters.fixation_end_time,
                                           self.__parameters.delta_time)
        ]
        return fixation_count_sorted_by_time

    @property
    @lru_cache(maxsize=1)
    def experiment_gaze_variances_sorted_by_time(self):
        experiment_variances_sorted_by_time: dict[int, list[float]] = {
            experiment.id: [
                screen_location_variance(screen_locations, average_screen_location_)
                for screen_locations, average_screen_location_ in zip(
                    self.__analyzed_experiments[experiment.id].screen_locations_sorted_by_time,
                    self.average_screen_locations_sorted_by_time
                )
            ]
            for experiment in self.__experiments
        }

        return experiment_variances_sorted_by_time

    @property
    @lru_cache(maxsize=1)
    def experiment_fixation_variances_sorted_by_time(self):
        experiment_variances_sorted_by_time: dict[int, list[float]] = {
            experiment.id: [
                screen_location_variance(screen_locations, average_screen_location_)
                for screen_locations, average_screen_location_ in zip(
                    self.__analyzed_experiments[experiment.id].fixation_locations_sorted_by_time,
                    self.average_fixation_locations_sorted_by_time
                )
            ]
            for experiment in self.__experiments
        }

        return experiment_variances_sorted_by_time

    @lru_cache(maxsize=1)
    def get_mean_number_of_blinks_and_duration(self):
        sum_blinks, sum_mean_duration = 0, 0
        for analyzed_experiment_id, analyzed_experiment in self.__analyzed_experiments.items():
            num_blinks, blinks_duration_mean = analyzed_experiment.get_num_of_blinks_and_mean_duration_in_video()
            sum_blinks += num_blinks
            sum_mean_duration += blinks_duration_mean

        return sum_blinks / len(self.__experiments), sum_mean_duration / len(self.__experiments)

    @lru_cache(maxsize=1)
    def get_mean_number_of_fixations_and_duration(self) -> tuple[float, float]:
        sum_fixations, sum_duration = 0, 0

        for analyzed_experiment_id, analyzed_experiment in self.__analyzed_experiments.items():
            num_fixations, duration_mean = analyzed_experiment.get_num_of_fixation_and_mean_duration_in_video()
            sum_fixations += num_fixations
            sum_duration += duration_mean

        fixations_mean: float = sum_fixations / len(self.__analyzed_experiments.keys())

        duration_mean: float = sum_duration / len(self.__analyzed_experiments.keys())

        return fixations_mean, duration_mean