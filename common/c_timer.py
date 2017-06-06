import time


class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.interval = -1

    def stop(self):
        self.interval = time.time() - self.start_time
        return self.check_interval()

    def check_interval(self):
        return self.interval
