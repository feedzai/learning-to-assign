from abc import ABC, abstractmethod


class AbstractExpert(ABC):

    @abstractmethod
    def predict(self):
        pass
