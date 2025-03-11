import time

class Timer:
    def __init__(self):
        self.startTime = None
        self.endTime = None
        self.lastElapsedTime = None
        pass

    def start(self):
        self.startTime = time.time()

    def stop(self):
        self.endTime = time.time()
        print("Elapsed time: ", self.endTime - self.startTime, " seconds...")

        self.lastElapsedTime = self.endTime - self.startTime
        self.startTime = None
        self.endTime = None
