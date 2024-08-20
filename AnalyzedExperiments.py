from __future__ import annotations
import math
import numpy as np
from configurations import MAX_DEVIATION
import statistics
from dataclasses import dataclass
import pandas
from experiment_types import Experiment, Eyesight, ScreenLocation
from functools import lru_cache


def average_screen_location(screen_locations: set[ScreenLocation],
                            max_deviation: float | None = None) -> ScreenLocation | None:
    """
    :param screen_locations: A set if ScreenLocation objects.
    :param max_deviation: The maximum number of standard deviations from the mean that a value can be.
    :return: The average of the screen locations that do not exceed the specified number of standard deviations from
     the mean.
    """
    if len(screen_locations) == 0:
        return None
    if max_deviation is None:
        x_mean = statistics.mean([screen_location.x for screen_location in screen_locations])
        y_mean = statistics.mean([screen_location.y for screen_location in screen_locations])
        return ScreenLocation(x_mean, y_mean)
    x_limited_stdev = limit_standard_deviation(
        [screen_location.x for screen_location in screen_locations],
        max_deviation
    )
    y_limited_stddev = limit_standard_deviation(
        [screen_location.y for screen_location in screen_locations],
        max_deviation
    )
    return ScreenLocation(statistics.mean(x_limited_stdev), statistics.mean(y_limited_stddev))


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


def get_mapped_gaze_start_time_to_end_time(experiments: list[Experiment]) -> tuple[int, int, int, int]:
    latest_first_saccade_time_ns: int = max(
        experiment.mapped_gaze_on_video.gaze["timestamp [ns]"].iloc[0]
        for experiment in experiments
    )

    earliest_first_saccade_time_ns: int = min(
        experiment.mapped_gaze_on_video.gaze["timestamp [ns]"].iloc[0]
        for experiment in experiments
    )

    earliest_last_saccade_time_ns: int = min(
        experiment.mapped_gaze_on_video.gaze["timestamp [ns]"].iloc[-1]
        for experiment in experiments
    )

    latest_last_saccade_time_ns: int = max(
        experiment.mapped_gaze_on_video.gaze["timestamp [ns]"].iloc[-1]
        for experiment in experiments
    )

    return latest_first_saccade_time_ns, earliest_last_saccade_time_ns


def limit_standard_deviation(values: list[int | float],
                             max_deviation: float) -> list[int | float]:
    """
    Filters the input list to include only values within a specified number of standard deviations from the mean.
    :param values: A list of numeric values.
    :param max_deviation: The maximum number of standard deviations from the mean that a value can be.
    :return: A list of values that do not exceed the specified number of standard deviations from the mean, the
    standard_deviation and mean
    """
    if len(values) < 2:
        return values

    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)

    lower_bound = mean - max_deviation * standard_deviation
    upper_bound = mean + max_deviation * standard_deviation

    filtered_values = [x for x in values if lower_bound <= x <= upper_bound]
    return filtered_values


def get_raw_data_fixation_start_time_to_end_time(experiments: list[Experiment]) -> tuple[int, int, int, int]:
    earliest_first_fixation_time_ns: int = min(
        experiment.mapped_gaze_on_video.fixations["start timestamp [ns]"].iloc[0]
        for experiment in experiments
    )

    latest_first_fixation_time_ns: int = max(
        experiment.mapped_gaze_on_video.fixations["start timestamp [ns]"].iloc[0]
        for experiment in experiments
    )

    earliest_last_fixation_time_ns: int = min(
        experiment.mapped_gaze_on_video.fixations["start timestamp [ns]"].iloc[-1]
        for experiment in experiments
    )

    latest_last_fixation_time_ns: int = max(
        experiment.mapped_gaze_on_video.fixations["start timestamp [ns]"].iloc[-1]
        for experiment in experiments
    )

    return (earliest_first_fixation_time_ns,
            latest_first_fixation_time_ns,
            earliest_last_fixation_time_ns,
            latest_last_fixation_time_ns)


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
    fixation_earliest_first_fixation_time_ns: int
    fixation_latest_first_fixation_time_ns: int
    fixation_earliest_last_fixation_time_ns: int
    fixation_latest_last_fixation_time_ns: int
    delta_time: int


def gaze_time_to_index(period_start_time: int,
                       parameters: AnalyzedExperimentsParameters) -> int:
    return (period_start_time - parameters.gaze_start_time) // parameters.delta_time


def fixation_time_to_index(period_start_time: int,
                           parameters: AnalyzedExperimentsParameters) -> int:
    return (period_start_time - parameters.fixation_earliest_first_fixation_time_ns) // parameters.delta_time


class AnalyzedExperiment:
    def __init__(self,
                 experiment: Experiment,
                 parameters: AnalyzedExperimentsParameters):

        self.__experiment: Experiment = experiment

        self.__parameters: AnalyzedExperimentsParameters = parameters

    @property
    def experiment(self):
        return self.__experiment

    @property
    def parameters(self):
        return self.__parameters

    @lru_cache(maxsize=80)
    def get_num_of_fixation_and_mean_duration_in_video(self) -> tuple[int, int]:
        """
        :return: The number of fixations between the first and second events and the mean duration of these fixations
                 (in milliseconds).
        """
        # The fixation df is already filtered within the specified time range of the video.
        fixation_df = self.experiment.mapped_gaze_on_video.fixations

        # Calculate the number of fixations and average fixation duration
        num_fixations = len(fixation_df)
        average_fixation_duration = fixation_df["duration [ms]"].mean()

        return num_fixations, average_fixation_duration

    @staticmethod
    def split_fixations_by_time(experiment: Experiment,
                                parameters: AnalyzedExperimentsParameters) -> list[set[ScreenLocation]]:
        fixations_locations_by_time: list[set[ScreenLocation]] = []

        for current_start_time in range(parameters.fixation_earliest_first_fixation_time_ns,
                                        parameters.fixation_latest_last_fixation_time_ns,
                                        parameters.delta_time):
            experiment_fixation = experiment.mapped_gaze_on_video.fixations
            fixations_in_time_period: pandas.DataFrame = experiment_fixation[
                (current_start_time <= experiment_fixation["start timestamp [ns]"])
                & (experiment_fixation["start timestamp [ns]"] < current_start_time + parameters.delta_time)
                ]

            locations_in_time_period = set(fixations_in_time_period.apply(
                lambda row: ScreenLocation(row["fixation position transf x[px]"],
                                           row["fixation position transf y[px]"],
                                           ),
                axis=1)
            )

            fixations_locations_by_time.append(locations_in_time_period)
        return fixations_locations_by_time

    @lru_cache(maxsize=80)
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

    @property
    @lru_cache(maxsize=80)
    def screen_locations_sorted_by_time(self) -> list[set[ScreenLocation]]:
        """
        :return: A list of sets containing ScreenLocations objects.
        Each set includes all gazes within the specified time period, and the list of sets is ordered by time.
        """

        screen_locations_by_time: list[set[ScreenLocation]] = []

        for current_start_time in range(self.parameters.gaze_start_time,
                                        self.parameters.gaze_end_time,
                                        self.parameters.delta_time):
            experiment_gaze = self.experiment.mapped_gaze_on_video.gaze
            gazes_in_time_period: pandas.DataFrame = experiment_gaze[
                (current_start_time <= experiment_gaze["timestamp [ns]"])
                & (experiment_gaze["timestamp [ns]"] < current_start_time + self.parameters.delta_time)
                ]

            locations_in_time_period = set(gazes_in_time_period.apply(
                lambda row: ScreenLocation(row["gaze position transf x [px]"],
                                           row["gaze position transf y [px]"]),
                axis=1)
            )
            screen_locations_by_time.append(locations_in_time_period)

        return screen_locations_by_time

    @property
    @lru_cache(maxsize=80)
    def fixation_locations_sorted_by_time(self):
        fixation_locations_sorted_by_time: list[set[ScreenLocation]] = (
            AnalyzedExperiment.split_fixations_by_time(self.experiment, self.parameters)
        )
        return fixation_locations_sorted_by_time

    @property
    @lru_cache(maxsize=80)
    def average_screen_locations(self) -> list[ScreenLocation]:
        """
        :return: The average gaze screen location. Each ScreenLocation object will contain the average indices
        for the specific time period, and the output list is sorted by the time period.
        """

        average_screen_locations: list[ScreenLocation] = [
            average_screen_location(screen_locations)
            for screen_locations in self.screen_locations_sorted_by_time
        ]
        return average_screen_locations

    @property
    @lru_cache(maxsize=80)
    def average_fixation_locations(self) -> list[ScreenLocation]:
        """
        :return: The average fixation screen location. Each ScreenLocation object will contain the average indices
        for the specific time period, and the output list is sorted by the time period.
        """
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

    def average_screen_locations_sorted_by_time(self, max_deviation: float | None = None) -> list[ScreenLocation]:
        """
        :param max_deviation: The maximum number of standard deviations from the mean that a value can be.
        :return: Average screen locations of all the experiments together.
        """
        average_screen_locations_sorted_by_time: list[ScreenLocation] = [
            average_screen_location(
                {
                    screen_location
                    for analyzed_experiment in self.__analyzed_experiments.values()
                    for screen_location in analyzed_experiment.screen_locations_sorted_by_time[
                    gaze_time_to_index(period_start_time, self.__parameters)
                ]
                },
                max_deviation=max_deviation
            )
            for period_start_time in range(self.__parameters.gaze_start_time,
                                           self.__parameters.gaze_end_time,
                                           self.__parameters.delta_time)
        ]

        return average_screen_locations_sorted_by_time

    def sem_gaze_locations_sorted_by_time(self, max_deviation: float | None = None):
        stdev_gaze_locations_sorted_by_time_x = []
        sem_gaze_locations_sorted_by_time_x = []
        stdev_gaze_locations_sorted_by_time_y = []
        sem_gaze_locations_sorted_by_time_y = []

        for period_start_time in range(self.__parameters.gaze_start_time,
                                       self.__parameters.gaze_end_time,
                                       self.__parameters.delta_time):
            gaze_locations_x_mean = []
            gaze_locations_y_mean = []
            for analyzed_experiment in self.__analyzed_experiments.values():
                gaze_locations_x_mean.append(statistics.mean([
                    screen_location.x
                    for screen_location in analyzed_experiment.screen_locations_sorted_by_time[
                        gaze_time_to_index(period_start_time, self.__parameters)
                    ]
                ]))

                gaze_locations_y_mean.append(statistics.mean([
                    screen_location.y
                    for screen_location in analyzed_experiment.screen_locations_sorted_by_time[
                        gaze_time_to_index(period_start_time, self.__parameters)
                    ]
                ]))

            if len(gaze_locations_x_mean) > 1:
                gaze_locations_filtered = limit_standard_deviation(gaze_locations_x_mean, max_deviation)
                deviation = statistics.stdev(gaze_locations_filtered)
                stdev_gaze_locations_sorted_by_time_x.append(deviation)
                sem_gaze_locations_sorted_by_time_x.append(deviation / math.sqrt(len(gaze_locations_x_mean)))
            else:
                stdev_gaze_locations_sorted_by_time_x.append(0)
                sem_gaze_locations_sorted_by_time_x.append(0)

            if len(gaze_locations_y_mean) > 1:
                gaze_locations_filtered = limit_standard_deviation(gaze_locations_y_mean, max_deviation)
                deviation = statistics.stdev(gaze_locations_filtered)
                stdev_gaze_locations_sorted_by_time_y.append(deviation)
                # print(gaze_locations_y_mean)
                # print(f"{deviation} / {math.sqrt(len(gaze_locations_y_mean))} = {deviation / math.sqrt(len(gaze_locations_y_mean))}")
                sem_gaze_locations_sorted_by_time_y.append(deviation / math.sqrt(len(gaze_locations_y_mean)))
            else:
                stdev_gaze_locations_sorted_by_time_y.append(0)
                sem_gaze_locations_sorted_by_time_y.append(0)

        return (stdev_gaze_locations_sorted_by_time_x,
                sem_gaze_locations_sorted_by_time_x,
                stdev_gaze_locations_sorted_by_time_y,
                sem_gaze_locations_sorted_by_time_y)

    @lru_cache(maxsize=3)
    def average_fixation_locations_sorted_by_time(self, max_deviation: float | None = None):
        average_fixation_locations_sorted_by_time: list[ScreenLocation] = [
            average_screen_location(
                {
                    screen_location
                    for analyzed_experiment in self.__analyzed_experiments.values()
                    for screen_location in analyzed_experiment.fixation_locations_sorted_by_time[
                    fixation_time_to_index(period_start_time, self.__parameters)
                ]
                },
                max_deviation=max_deviation
            )
            for period_start_time in range(self.__parameters.fixation_earliest_first_fixation_time_ns,
                                           self.__parameters.fixation_latest_last_fixation_time_ns,
                                           self.__parameters.delta_time)
        ]

        return average_fixation_locations_sorted_by_time

    def sem_fixation_locations_sorted_by_time(self, max_deviation: float | None = None):
        sem_fixation_locations_sorted_by_time_x = []
        sem_fixation_locations_sorted_by_time_y = []

        for period_start_time in range(self.__parameters.fixation_earliest_first_fixation_time_ns,
                                       self.__parameters.fixation_latest_last_fixation_time_ns,
                                       self.__parameters.delta_time):

            fixation_locations_x = [
                screen_location.x
                for analyzed_experiment in self.__analyzed_experiments.values()
                for screen_location in analyzed_experiment.fixation_locations_sorted_by_time[
                    fixation_time_to_index(period_start_time, self.__parameters)
                ]
            ]

            fixation_locations_y = [
                screen_location.y
                for analyzed_experiment in self.__analyzed_experiments.values()
                for screen_location in analyzed_experiment.fixation_locations_sorted_by_time[
                    fixation_time_to_index(period_start_time, self.__parameters)
                ]
            ]

            if len(fixation_locations_x) > 1:
                fixation_locations_filtered = limit_standard_deviation(fixation_locations_x, max_deviation)
                deviation = statistics.stdev(fixation_locations_filtered)
                sem_fixation_locations_sorted_by_time_x.append(deviation / math.sqrt(len(fixation_locations_x)))
            else:
                sem_fixation_locations_sorted_by_time_x.append(0)

            if len(fixation_locations_y) > 1:
                fixation_locations_filtered = limit_standard_deviation(fixation_locations_y, max_deviation)
                deviation = statistics.stdev(fixation_locations_filtered)
                sem_fixation_locations_sorted_by_time_y.append(deviation / math.sqrt(len(fixation_locations_y)))
            else:
                sem_fixation_locations_sorted_by_time_y.append(0)

        return sem_fixation_locations_sorted_by_time_x, sem_fixation_locations_sorted_by_time_y

    @property
    def fixation_count_sorted_by_time(self) -> list[list[int]]:
        fixation_count_sorted_by_time: list[list[int]] = [
            [
                len(analyzed_experiment.fixation_locations_sorted_by_time[
                        fixation_time_to_index(period_start_time, self.__parameters)
                    ])
                for analyzed_experiment in self.__analyzed_experiments.values()
            ]
            for period_start_time in range(self.__parameters.fixation_earliest_first_fixation_time_ns,
                                           self.__parameters.fixation_latest_last_fixation_time_ns,
                                           self.__parameters.delta_time)
        ]
        return fixation_count_sorted_by_time

    @property
    @lru_cache(maxsize=3)
    def experiment_gaze_variances_sorted_by_time(self):
        """
        :return: This function return the gaze location variance sorted by time.
        it is calculated using the average gaze of all the experiments in the class divided by time periods.
        """
        experiment_variances_sorted_by_time: dict[int, list[float]] = {
            experiment.id: [
                screen_location_variance(screen_locations, average_screen_location_)
                for screen_locations, average_screen_location_ in zip(
                    self.__analyzed_experiments[experiment.id].screen_locations_sorted_by_time,
                    self.average_screen_locations_sorted_by_time()
                )
            ]
            for experiment in self.__experiments
        }

        return experiment_variances_sorted_by_time

    @property
    @lru_cache(maxsize=3)
    def experiment_fixation_variances_sorted_by_time(self):
        experiment_variances_sorted_by_time: dict[int, list[float]] = {
            experiment.id: [
                screen_location_variance(screen_locations, average_screen_location_)
                for screen_locations, average_screen_location_ in zip(
                    self.__analyzed_experiments[experiment.id].fixation_locations_sorted_by_time,
                    self.average_fixation_locations_sorted_by_time()
                )
            ]
            for experiment in self.__experiments
        }

        return experiment_variances_sorted_by_time

    # @lru_cache(maxsize=3)
    # def get_mean_number_of_blinks_and_duration(self):
    #     sum_blinks, sum_mean_duration = 0, 0
    #     for analyzed_experiment_id, analyzed_experiment in self.__analyzed_experiments.items():
    #         num_blinks, blinks_duration_mean = analyzed_experiment.get_num_of_blinks_and_mean_duration_in_video()
    #         sum_blinks += num_blinks
    #         sum_mean_duration += blinks_duration_mean

        # return sum_blinks / len(self.__experiments), sum_mean_duration / len(self.__experiments)

    def get_mean_number_of_blinks_and_duration(self) -> tuple[float, float, float, float]:
        blink_counts = []
        blink_durations = []

        for analyzed_experiment_id, analyzed_experiment in self.__analyzed_experiments.items():
            num_blinks, blinks_duration_mean = analyzed_experiment.get_num_of_blinks_and_mean_duration_in_video()
            blink_counts.append(num_blinks)
            blink_durations.append(blinks_duration_mean)

        blink_counts_mean: float = statistics.mean(blink_counts)
        duration_mean: float = statistics.mean(blink_durations)

        blinks_sem: float = statistics.stdev(blink_counts) / (len(blink_counts) ** 0.5)
        duration_sem: float = statistics.stdev(blink_durations) / (len(blink_durations) ** 0.5)

        return blink_counts_mean, duration_mean, blinks_sem, duration_sem

    @lru_cache(maxsize=3)
    def get_mean_fixations_count_and_duration(self) -> tuple[float, ...]:
        #sum_fixations, sum_duration = 0, 0
        fixations_counts = []
        durations = []

        for analyzed_experiment_id, analyzed_experiment in self.__analyzed_experiments.items():
            num_fixations, duration_mean = analyzed_experiment.get_num_of_fixation_and_mean_duration_in_video()
            durations.append(duration_mean)

            fixations_counts.append(num_fixations)

        fixations_mean: float = statistics.mean(fixations_counts)
        duration_mean: float = statistics.mean(durations)

        fixations_sem: float = statistics.stdev(fixations_counts) / (len(fixations_counts) ** 0.5)
        duration_sem: float = statistics.stdev(durations) / (len(durations) ** 0.5)

        return fixations_mean, duration_mean, fixations_sem, duration_sem

    @lru_cache(maxsize=3)
    def get_list_of_mean_fixations_count_and_duration_per_experiment(self) -> tuple[list[int], list[int]]:
        fixations_list = []
        duration_list = []

        for analyzed_experiment_id, analyzed_experiment in self.__analyzed_experiments.items():
            num_fixations, duration_mean = analyzed_experiment.get_num_of_fixation_and_mean_duration_in_video()
            fixations_list.append(num_fixations)
            duration_list.append(duration_mean)

        return fixations_list, duration_list