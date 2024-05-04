from typing import List, Dict, Optional
from collections import OrderedDict


def add_times(times: OrderedDict[str, float], times_to_add: OrderedDict[str, float]):
    for key, value in times_to_add.items():
        if key not in times:
            times[key] = 0.0
        times[key] += value


def add_counts(counts: OrderedDict[str, int], counts_to_add: OrderedDict[str, int]):
    for key, value in counts_to_add.items():
        if key not in counts:
            counts[key] = 0
        counts[key] += value


def init_times(time_names: List[str]) -> OrderedDict[str, float]:
    times: OrderedDict[str, float] = OrderedDict()
    for time_name in time_names:
        times[time_name] = 0.0
    return times


def init_counts(time_names: List[str]) -> OrderedDict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for time_name in time_names:
        counts[time_name] = 0
    return counts


class Times:

    def __init__(self, time_names: Optional[List[str]] = None):
        if time_names is None:
            time_names = []

        self.times: OrderedDict[str, float] = init_times(time_names)
        self.counts: OrderedDict[str, int] = init_counts(time_names)

        self.sub_times: Dict[str, Times] = dict()
        self.sub_counts: Dict[str, int] = dict()

    def record_time(self, time_name: str, time_elapsed: float, path: Optional[List[str]] = None):
        if (path is not None) and (len(path) > 0):
            path_0: str = path.pop(0)
            if path_0 not in self.sub_times:
                self.sub_times[path_0] = Times()
                self.sub_counts[path_0] = 0

            self.sub_times[path_0].record_time(time_name, time_elapsed, path=path)
            self.sub_counts[path_0] += 1
        else:
            if time_name not in self.times.keys():
                self.times[time_name] = 0
                self.counts[time_name] = 0

            self.times[time_name] += time_elapsed
            self.counts[time_name] += 1

    def add_times(self, time: 'Times', path: Optional[List[str]] = None):
        if (path is not None) and (len(path) > 0):
            path_0: str = path.pop(0)
            if path_0 not in self.sub_times:
                self.sub_times[path_0] = Times()
                self.sub_counts[path_0] = 0
            self.sub_times[path_0].add_times(time, path=path)
            self.sub_counts[path_0] += sum(time.counts.values())
        else:
            add_times(self.times, time.times)
            add_counts(self.counts, time.counts)
            for sub_time_name in time.sub_times.keys():
                if sub_time_name not in self.sub_times.keys():
                    self.sub_times[sub_time_name] = Times()
                    self.sub_counts[sub_time_name] = 0

                self.sub_times[sub_time_name].add_times(time.sub_times[sub_time_name])
                self.sub_counts[sub_time_name] += sum(time.sub_times[sub_time_name].counts.values())

    def reset_times(self):
        for key in self.times.keys():
            self.times[key] = 0.0
            self.counts[key] = 0

        for sub_time in self.sub_times.values():
            sub_time.reset_times()

    def get_total_time(self) -> float:
        time_tot: float = 0.0
        for time_elapsed in self.times.values():
            time_tot += time_elapsed

        for sub_time in self.sub_times.values():
            time_tot += sub_time.get_total_time()

        return time_tot

    def get_time_str(self, prefix: str = "") -> str:
        time_str_l: List[str] = ["%s: %f" % (key, val) for key, val in self.times.items()]
        sub_time_str_l: List[str] = ["->%s: %f" % (key, sub_time.get_total_time())
                                     for key, sub_time in self.sub_times.items()]

        time_str: str = ", ".join(time_str_l + sub_time_str_l + ["Tot: %f" % self.get_total_time()])

        prefix_new: str = f"\t{prefix}"
        for key, sub_time in self.sub_times.items():
            time_str = f"{time_str}\n{prefix_new}({key}): {sub_time.get_time_str(prefix=prefix_new)}"

        return time_str

    def __str__(self):
        return self.get_time_str()

    def __repr__(self):
        return self.__str__()
