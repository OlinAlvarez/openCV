from abc import ABCMeta,abstractmethod
#Interface
class Task(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_directions(self):
        pass
    @abstractmethod
    def isTaskComplete(self):
        pass
