import os
import json
import uuid
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime

# Path to store experiences
DATA_PATH = Path.home() / ".autogoal" / "experience_store"


class Experience:
    def __init__(
        self,
        model_class_name: str,
        finetuning_method: str,
        finetuning_parameters: Dict[str, Any],
        dataset_features: np.ndarray,
        system_features: np.ndarray,
        dataset_feature_extractor_name: str,
        system_feature_extractor_name: str,
        timestamp: str,
        cross_val_steps: Optional[int] = None,
        accuracy: Optional[float] = None,
        f1: Optional[float] = None,
        evaluation_time: Optional[float] = None,
        error: Optional[str] = None,
    ):
        self.model_class_name = model_class_name
        self.finetuning_method = finetuning_method
        self.finetuning_parameters = finetuning_parameters
        self.dataset_features = dataset_features
        self.system_features = system_features
        self.dataset_feature_extractor_name = dataset_feature_extractor_name
        self.system_feature_extractor_name = system_feature_extractor_name
        self.timestamp = timestamp
        self.accuracy = accuracy
        self.cross_val_steps = cross_val_steps
        self.f1 = f1
        self.evaluation_time = evaluation_time
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_class_name': self.model_class_name,
            'finetuning_method': self.finetuning_method,
            'finetuning_parameters': self.finetuning_parameters,
            'dataset_features': self.dataset_features.tolist(),
            'system_features': self.system_features.tolist(),
            'dataset_feature_extractor_name': self.dataset_feature_extractor_name,
            'system_feature_extractor_name': self.system_feature_extractor_name,
            'timestamp': self.timestamp,
            'accuracy': self.accuracy,
            'cross_val_steps': self.cross_val_steps,
            'f1': self.f1,
            'evaluation_time': self.evaluation_time,
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        return cls(
            model_class_name=data['model_class_name'],
            finetuning_method=data['finetuning_method'],
            finetuning_parameters=data['finetuning_parameters'],
            dataset_features=np.array(data['dataset_features']),
            system_features=np.array(data['system_features']),
            dataset_feature_extractor_name=data.get('dataset_feature_extractor_name', 'Unknown'),
            system_feature_extractor_name=data.get('system_feature_extractor_name', 'Unknown'),
            timestamp=data['timestamp'],
            accuracy=data.get('accuracy'),
            cross_val_steps=data.get('cross_val_steps'),
            f1=data.get('f1'),
            evaluation_time=data.get('evaluation_time'),
            error=data.get('error'),
        )


class ExperienceStore:
    DATA_PATH = DATA_PATH

    @staticmethod
    def save_experience(experience: Experience):
        """
        Saves an experience to disk as a JSON file, grouping experiences by date.
        """
        # Ensure the base data directory exists
        ExperienceStore.DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Parse the timestamp to get the date (YYYY-MM-DD)
        timestamp = experience.timestamp
        try:
            date_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            date_str = date_obj.strftime("%Y-%m-%d")
            time_str = date_obj.strftime("%H:%M:%S")
        except ValueError:
            # Handle different timestamp formats or default to today's date
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Create a directory for the date
        date_dir = ExperienceStore.DATA_PATH / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        experience_id = str(uuid.uuid4())
        file_path = date_dir / f"{time_str}-{experience_id}.json"

        # Serialize the experience to JSON
        exp_dict = experience.to_dict()
        if 'device' in exp_dict['finetuning_parameters']:
            exp_dict['finetuning_parameters'].pop('device')
        
        with open(file_path, 'w') as f:
            json.dump(exp_dict, f, indent=4)

    @staticmethod
    def load_all_experiences(from_date: Optional[Union[str, date]] = None, to_date: Optional[Union[str, date]] = None) -> List[Experience]:
        """
        Loads all experiences from disk, traversing all date folders, with optional date filtering.

        Args:
            from_date (Optional[Union[str, date]]): The start date in "YYYY-MM-DD" format or a date object.
            to_date (Optional[Union[str, date]]): The end date in "YYYY-MM-DD" format or a date object.

        Returns:
            A list of Experience instances.
        """
        experiences = []
        if not ExperienceStore.DATA_PATH.exists():
            # No experiences saved yet
            return experiences

        # Parse the date filters if provided
        if isinstance(from_date, str):
            from_date_obj = datetime.strptime(from_date, "%Y-%m-%d").date()
        elif isinstance(from_date, date):
            from_date_obj = from_date
        else:
            from_date_obj = None

        if isinstance(to_date, str):
            to_date_obj = datetime.strptime(to_date, "%Y-%m-%d").date()
        elif isinstance(to_date, date):
            to_date_obj = to_date
        else:
            to_date_obj = None

        # Traverse all date directories
        for date_dir in ExperienceStore.DATA_PATH.iterdir():
            if date_dir.is_dir():
                # Parse the directory name as a date
                try:
                    dir_date_obj = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                except ValueError:
                    # Skip directories that do not match the date format
                    continue

                # Apply date filtering
                if from_date_obj and dir_date_obj < from_date_obj:
                    continue
                if to_date_obj and dir_date_obj > to_date_obj:
                    continue

                # Iterate over all JSON files in the date directory
                for file_path in date_dir.glob('*.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        experience = Experience.from_dict(data)
                        experiences.append(experience)
        return experiences