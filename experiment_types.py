from dataclasses import dataclass
from enum import Enum
import pandas as pd
import os
import numpy as np
import math
import json


class Eyesight(Enum):
    def __str__(self):
        return "good" if self.value == 1 else "bad"

    GOOD = 1
    BAD = 2


@dataclass(frozen=True)
class ScreenLocation:
    x: float
    y: float

    def distance(self, other: "ScreenLocation") -> float:
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))


@dataclass
class RawData:
    blinks: pd.DataFrame
    events: pd.DataFrame
    fixations: pd.DataFrame
    gaze: pd.DataFrame
    imu: pd.DataFrame
    world_timestamps: pd.DataFrame


@dataclass
class ReferenceData:
    fixations: pd.DataFrame
    gaze: pd.DataFrame
    sections: pd.DataFrame


@dataclass
class MappedGazeOnVideo:
    gaze: pd.DataFrame
    fixations: pd.DataFrame


@dataclass
class ExperimentInput:
    eyesight: Eyesight
    raw_data_dir_path: str
    reference_data_dir_path: str
    mapped_gaze_on_video_dir_path: str


@dataclass
class Info:
    recording_id: str


class Experiment:
    __available_id: int = 0

    def __init__(self,
                 experiment_input: ExperimentInput):
        Experiment.__available_id += 1

        # todo fix! theres an actual id.
        self.__id: int = Experiment.__available_id
        self.__eyesight: Eyesight = experiment_input.eyesight
        self.__raw_data: RawData = self.__parse_raw_data_dir(experiment_input.raw_data_dir_path)
        self.__info: Info = Experiment.__parse_info_file(
            os.path.join(experiment_input.raw_data_dir_path, "info.json")
        )
        self.__reference_data: ReferenceData = self.__parse_reference_image_data(
            experiment_input.reference_data_dir_path
        )
        self.__mapped_gaze_on_video: MappedGazeOnVideo = self.__parse_mapped_gaze_on_video(
            experiment_input.mapped_gaze_on_video_dir_path
        )

    @property
    def id(self):
        return self.__id

    @property
    def eyesight(self):
        return self.__eyesight

    @property
    def raw_data(self):
        return self.__raw_data

    @property
    def reference_data(self):
        return self.__reference_data

    @property
    def mapped_gaze_on_video(self):
        return self.__mapped_gaze_on_video

    @property
    def video_start_timestamp(self):
        return self.__video_start_timestamp

    @property
    def video_end_timestamp(self):
        return self.__video_end_timestamp

    @property
    def recording_id(self):
        return self.__info.recording_id

    @staticmethod
    def __parse_info_file(info_file_path: str):
        with open(info_file_path) as f:
            data = json.load(f)

        # Extract the recording_id
        recording_id = data["recording_id"]

        print("Recording ID:", recording_id)

        return Info(recording_id)

    def __parse_raw_data_dir(self, raw_data_dir_path: str) -> RawData:
        """
        :param raw_data_dir_path: The raw data dir as it was downloaded from Pupil Cloud.
        :return: The parsed data from the directory's files.
        """
        events_headers: list[str] = ["recording id", "timestamp [ns]", "name", "type"]
        events_path: str = rf"{raw_data_dir_path}\events.csv"
        events_df: pd.DataFrame = pd.read_csv(events_path, names=events_headers, skiprows=1)

        self.__video_start_timestamp = events_df.loc[
            events_df["name"] == "start.video", "timestamp [ns]"
        ].values[0]
        self.__video_end_timestamp = events_df.loc[
            events_df["name"] == "end.video", "timestamp [ns]"
        ].values[0]
        blinks_headers: list[str] = ["section id", "recording id", "blink id",
                                     "start timestamp [ns]", "end timestamp [ns]", "duration [ms]"]
        blinks_path: str = rf"{raw_data_dir_path}\blinks.csv"
        blinks_df: pd.DataFrame = pd.read_csv(blinks_path, names=blinks_headers, skiprows=1)
        blink_int_columns: list[str] = ["start timestamp [ns]", "end timestamp [ns]", "duration [ms]"]
        blinks_df[blink_int_columns] = blinks_df[blink_int_columns].astype(np.int64)

        fixations_headers: list[str] = ["section id", "recording id", "fixation id", "start timestamp [ns]",
                                        "end timestamp [ns]", "duration [ms]", "fixation x [px]", "fixation y [px]",
                                        "azimuth [deg]", "elevation [deg]"]
        fixations_path: str = rf"{raw_data_dir_path}\fixations.csv"
        fixations_float_columns: list[str] = ["fixation x [px]", "fixation y [px]"]
        fixations_int_columns: list[str] = ["start timestamp [ns]", "end timestamp [ns]", "duration [ms]"]
        fixations_df = self.__create_basic_df_with_timestamps(
            fixations_headers, ["start timestamp [ns]", "end timestamp [ns]"], fixations_float_columns,
            fixations_int_columns, fixations_path)

        gaze_headers: list[str] = [
            "section id", "recording id", "timestamp [ns]", "gaze x [px]", "gaze y [px]", "worn", "fixation id",
            "blink id", "azimuth [deg]", "elevation [deg]"
        ]
        gaze_path: str = rf"{raw_data_dir_path}\gaze.csv"
        gaze_df: pd.DataFrame = self.__create_basic_df_with_timestamps(
            gaze_headers, ["timestamp [ns]"], [],
            [], gaze_path)

        imu_headers: list[str] = ["section id", "recording id", "timestamp [ns]", "gyro x [deg/s]", "gyro y [deg/s]",
                                  "gyro z [deg/s]", "acceleration x [G]", "acceleration y [G]", "acceleration z [G]",
                                  "roll [deg]", "pitch [deg]", "yaw [deg]", "quaternion w", "quaternion x",
                                  "quaternion y",
                                  "quaternion z"]
        imu_path: str = rf"{raw_data_dir_path}\imu.csv"
        imu_df: pd.DataFrame = self.__create_basic_df_with_timestamps(
            imu_headers, ["timestamp [ns]"], [],
            [], imu_path)

        world_timestamps_headers: list[str] = ["section id", "recording id", "timestamp [ns]"]
        world_timestamps_path: str = rf"{raw_data_dir_path}\world_timestamps.csv"
        world_timestamps_df: pd.DataFrame = pd.read_csv(world_timestamps_path, names=world_timestamps_headers,
                                                        skiprows=1)

        return RawData(blinks_df, events_df, fixations_df, gaze_df, imu_df, world_timestamps_df)

    def __parse_reference_image_data(self, reference_data_dir_path: str) -> ReferenceData:
        fixations_headers: list[str] = ["section id", "recording id", "fixation id", "start timestamp [ns]",
                                        "end timestamp [ns]", "duration [ms]", "fixation detected in reference image",
                                        "fixation x [px]", "fixation y [px]"]
        fixations_path: str = rf"{reference_data_dir_path}\fixations.csv"
        fixations_df: pd.DataFrame = self.__create_basic_df_with_timestamps(
            fixations_headers, ["start timestamp [ns]", "end timestamp [ns]"], [],
            [], fixations_path)
        # Filter fixation data within the specified time range
        fixations_df = fixations_df[fixations_df["recording id"] == self.recording_id]

        gaze_headers: list[str] = ["section id", "recording id", "timestamp [ns]", " gaze detected in reference image",
                                   "gaze position in reference image x [px]", "gaze position in reference image y [px]",
                                   "fixation id"]
        gaze_path: str = rf"{reference_data_dir_path}\gaze.csv"
        gaze_df: pd.DataFrame = self.__create_basic_df_with_timestamps(
            gaze_headers, ["timestamp [ns]"], [], [], gaze_path)

        # Note that the timestamp in section_df are raw and not minus the video start event.
        sections_headers: list[str] = [
            "section id", "recording id", "recording name", "wearer id", "wearer name", "section start time [ns]",
            "section end time [ns]", "start event name", "end event name"]
        sections_path: str = rf"{reference_data_dir_path}\sections.csv"
        sections_df: pd.DataFrame = pd.read_csv(sections_path, names=sections_headers, skiprows=1)

        return ReferenceData(fixations_df, gaze_df, sections_df)

    def __parse_mapped_gaze_on_video(self, mapped_gaze_on_video_path: str) -> MappedGazeOnVideo:
        gaze_headers: list[str] = [
            "section id", "recording id", "timestamp [ns]", "gaze detected in reference image",
            "gaze position in reference image x [px]", "gaze position in reference image y [px]", "fixation id",
            "recording name", "wearer id", "wearer name", "section start time [ns]", "section end time [ns]",
            "start event name", "end event name", "gaze position transf x [px]", "gaze position transf y [px]"
        ]
        gaze_float_columns: list[str] = [
            "gaze position in reference image x [px]", "gaze position in reference image y [px]",
            "gaze position transf x [px]", "gaze position transf y [px]"
        ]
        gaze_int_columns: list[str] = ["timestamp [ns]", "section start time [ns]", "section end time [ns]"]
        gaze_path: str = os.path.join(mapped_gaze_on_video_path, "gaze.csv")
        gaze_df: pd.DataFrame = self.__create_basic_df_with_timestamps(
            gaze_headers, ["timestamp [ns]"], gaze_float_columns, gaze_int_columns, gaze_path
        )

        fixations_headers: list[str] = [
            "section id", "recording id", "fixation id", "start timestamp [ns]", "end timestamp [ns]", "duration [ms]",
            "fixation detected in reference image", "fixation position in reference image x[px]",
            "fixation position in reference image y[px]", "recording name", "wearer id",
            "wearer name", "section start time[ns]", "section end time[ns]", "start event name", "end event name",
            "fixation position transf x[px]", "fixation position transf y[px]"
        ]
        fixations_float_columns: list[str] = [
            "fixation position in reference image x[px]", "fixation position in reference image y[px]",
            "fixation position transf x[px]", "fixation position transf y[px]"
        ]
        fixations_int_columns: list[str] = ["start timestamp [ns]", "end timestamp [ns]", "duration [ms]"]

        fixations_path: str = os.path.join(mapped_gaze_on_video_path, "fixations.csv")
        fixations_df = self.__create_basic_df_with_timestamps(
            fixations_headers, ["start timestamp [ns]", "end timestamp [ns]"],
            fixations_float_columns, fixations_int_columns, fixations_path
        )

        return MappedGazeOnVideo(gaze_df, fixations_df)

    def __adjust_timestamps_to_match_video(self,
                                           df: pd.DataFrame,
                                           timestamp_columns: list[str]) -> pd.DataFrame:
        # Filter data within the specified time range
        condition = True
        for timestamp_column in timestamp_columns:
            condition &= ((df[timestamp_column] >= self.__video_start_timestamp) &
                          (df[timestamp_column] <= self.__video_end_timestamp))
        df = df[condition]
        for timestamp_column in timestamp_columns:
            df.loc[:, timestamp_column] -= self.__video_start_timestamp

        return df

    def __create_basic_df_with_timestamps(self, headers: list[str],
                                          timestamp_columns: list[str],
                                          float_columns: list[str],
                                          int_columns: list[str],
                                          path: str,
                                          dtypes=None) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(path, names=headers, skiprows=1, sep=",", dtype=dtypes)
        df[float_columns] = df[float_columns].astype(float)
        df[int_columns] = df[int_columns].astype(np.int64)
        df = self.__adjust_timestamps_to_match_video(df, timestamp_columns)
        return df
