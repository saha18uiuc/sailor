from abc import ABC, abstractmethod


class BaselinePlanner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_sorted_plans(self):
        pass
