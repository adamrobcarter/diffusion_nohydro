import time

class LoopTiming:
    start_times = {}
    accumulated_times = {}

    def __init__(self):
        self.start_time = time.time()

    def start(self, name):
        assert name not in self.start_times, f"'{name}' already started."

        self.start_times[name] = time.time()

    def stop(self, name):
        assert name in self.start_times, f"'{name}' not started."

        elapsed_time = time.time() - self.start_times[name]
        self.accumulated_times[name] = self.accumulated_times.get(name, 0) + elapsed_time
        del self.start_times[name]

    def print_results(self):
        total_time = time.time() - self.start_time
        
        print('Loop timing results:')
        for name, elapsed_time in self.accumulated_times.items():
            percentage = (elapsed_time / total_time) * 100
            print(f"  {name}: {percentage:.0f}%")