from abc import ABC, abstractmethod

class SprayStrategy(ABC):
    @abstractmethod
    def calculate_spray_time(self, density_percent: float, max_ms: int = 2000) -> int:
        pass

class LowDensitySprayStrategy(SprayStrategy):
    def calculate_spray_time(self, density_percent: float, max_ms: int = 2000) -> int:
        if density_percent <= 0:
            return 0
        spray_ms = int(300 + (density_percent / 20.0) * (max_ms - 300))
        return max(300, min(spray_ms, 800)) # Cap at 800ms for low density

class MediumDensitySprayStrategy(SprayStrategy):
    def calculate_spray_time(self, density_percent: float, max_ms: int = 2000) -> int:
        spray_ms = int(300 + (density_percent / 20.0) * (max_ms - 300))
        return max(800, min(spray_ms, 1500)) # Range 800-1500ms for medium density

class HighDensitySprayStrategy(SprayStrategy):
    def calculate_spray_time(self, density_percent: float, max_ms: int = 2000) -> int:
        spray_ms = int(300 + (density_percent / 20.0) * (max_ms - 300))
        return max(1500, min(spray_ms, max_ms)) # Range 1500-max_ms for high density

class SprayContext:
    def __init__(self, strategy: SprayStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SprayStrategy):
        self._strategy = strategy

    def get_spray_time(self, density_percent: float, max_ms: int = 2000) -> int:
        return self._strategy.calculate_spray_time(density_percent, max_ms)

def get_strategy_for_density(density_percent: float) -> SprayStrategy:
    if density_percent < 5:
        return LowDensitySprayStrategy()
    elif density_percent < 15:
        return MediumDensitySprayStrategy()
    else:
        return HighDensitySprayStrategy()
