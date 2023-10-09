"""
Timers
"""
import time
from collections import defaultdict


class Timer(object):

    def __init__(self):
        self.reset()
    
    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def total(self):
        """ Return the total amount of time. """
        return self.total_time
    
    def avg(self):
        """ Return the average amount of time. """
        return self.total_time / float(self.calls)
    
    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    
class Timers(object):

    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def print(self, key=None):
        if key is None:
            # print all time 
            for k, v in self.timers.items():
                print("Average time for {:}: {:}".format(k, v.avg()))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))
    
    def get_avg(self, key):
        return self.timers[key].avg()
    

class NullTimer(object):
    """ NullTimer does nothing when its methods are called. """
    def __init__(self):
        pass

    def tic(self, *args):
        pass

    def toc(self, *args):
        pass

    def print(self, *args):
        pass
    