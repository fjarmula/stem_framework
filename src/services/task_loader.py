import yaml
import os
from src.config import config


class TaskLoader:
    def __init__(self, file_path=config["experiments"]["dir"]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Task file not found: {file_path}")
        with open(file_path, "r") as f:
            self.data = yaml.safe_load(f)

    @property
    def evolution_tasks(self):
        return self.data.get("evolution_set", [])

    @property
    def validation_tasks(self):
        return self.data.get("validation_set", [])
