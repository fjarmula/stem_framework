import os
from typing import Optional


class PromptManager:
    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir:
            self.prompts_dir = prompts_dir
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.prompts_dir = os.path.join(base_dir, "prompts")

    def get_prompt(self, filename: str, **kwargs) -> str:
        path = os.path.join(self.prompts_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt template not found at {path}")
        with open(path, "r") as f:
            template = f.read().strip()
        return template.format(**kwargs)
