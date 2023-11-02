import pandas as pd
from dataclasses import dataclass
from enum import Enum


class Eyesight(Enum):
    def __str__(self):
        return "good" if self.value == 1 else "bad"
    GOOD = 1
    BAD = 2


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


@dataclass
class Experiment:
    eyesight: Eyesight
    raw_data: RawData
    reference_data: ReferenceData
    mapped_gaze_on_video: MappedGazeOnVideo


@dataclass
class ExperimentInput:
    eyesight: Eyesight
    raw_data_dir_path: str
    reference_data_dir_path: str
    mapped_gaze_on_video_dir_path: str
