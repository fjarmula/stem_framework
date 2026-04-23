from pydantic import BaseModel


class ExperimentMetrics(BaseModel):
    stem_attempts: int = 0
    stem_passes: int = 0
    mature_attempts: int = 0
    mature_passes: int = 0

    @property
    def stem_pass_rate(self) -> float:
        return self.stem_passes / self.stem_attempts if self.stem_attempts else 0.0

    @property
    def mature_pass_rate(self) -> float:
        return self.mature_passes / self.mature_attempts if self.mature_attempts else 0.0

    def record(self, success: bool, is_stem: bool) -> None:
        if is_stem:
            self.stem_attempts += 1
            self.stem_passes += int(success)
        else:
            self.mature_attempts += 1
            self.mature_passes += int(success)

    def print_summary(self) -> None:
        print("\n" + "=" * 40)
        print("PASS RATE")
        print("=" * 40)
        print(f"Stem   {self.stem_passes}/{self.stem_attempts} attempts  →  {self.stem_pass_rate:.0%}")
        print(f"Mature {self.mature_passes}/{self.mature_attempts} attempts  →  {self.mature_pass_rate:.0%}")
        print("=" * 40)
