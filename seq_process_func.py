# coding : utf - 8
import gc
import pandas as pds
from tqdm import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
pds.set_option('display.max_columns', None)
work_path = "/home/lizhe/code/user_profile/"


def ts2second(x):
    return x // 1000


def ts2minutes(x):
    return x // 1000 // 60


def seq2topology(x: pds.DataFrame):
    events = x.event_id.values
    times = x.receive_time.values
    slow_point, fast_point = 0, 1
    while fast_point < len(events):
        if events[fast_point] != events[slow_point]:
            slow_point += 1
            events[slow_point] = events[fast_point]
            times[slow_point] = times[fast_point]
        else:
            fast_point += 1
    event_str = "_".join([str(e) for e in events[:slow_point + 1]])
    receive_str = "_".join([str(t) for t in times[:slow_point + 1]])
    return pds.DataFrame({"event_id": event_str, "receive_time": receive_str}, index=[0])


# Step 2: Split and Denoising
def split_denoise_apply(x: pds.DataFrame):
    ids = x.event_id[1:-1].split("_")
    ts = x.receive_time[1:-1].split("_")
    ids = ["0"] + ids + ["1"]
    ts = [ts[0]] + ts + [ts[-1]]
    assert len(ids) == len(ts), "Lens not match!"
    current_start_index = 0
    num_map = {}
    # Dynamic drop max duplicated subseqs
    index = 0

    res_ids, res_ts = [], []

    while True:
        if index >= len(ids):
            break
        if ids[index] == "0" or ids[index] == "1":
            if index > current_start_index + 1:
                res_ids.append("_".join(ids[current_start_index + 1:index]))
                res_ts.append("_".join(ts[current_start_index + 1:index]))
            current_start_index = index
            num_map = {ids[index]: index}
        # ONly drop duplicated subseq in a small time fraction 5min
        elif ids[index] not in num_map.keys():
            num_map[ids[index]] = index
        else:
            last_time = ts[num_map[ids[index]]]
            index_time = ts[index]
            # if float(index_time) - float(last_time) > 5:
            #     index += 1
            #     continue
            if len(ids) - index < index - num_map[ids[index]]:
                index += 1
                continue
            else:
                last_num_index = num_map[ids[index]]
                subseq_lens = index - last_num_index
                if ids[last_num_index:index] == ids[index:index + subseq_lens]:
                    del ids[last_num_index:index]
                    del ts[last_num_index:index]
                    num_map = {}
                    index = last_num_index
                    continue
                else:
                    num_map[ids[index]] = index
        index += 1
    x.event_id = res_ids
    x.receive_time = res_ts
    return x

