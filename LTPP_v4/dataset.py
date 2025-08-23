# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from constants import mental_predicate_set, action_predicate_set, PAD

class EventData(Dataset):
    """Event stream dataset.

    Each instance: list of dicts {'time_since_start', 'type_event'}.
    The dataset groups mental events by timestamp (as lists) and keeps actions
    as a time-ordered sequence.
    """

    def __init__(self, data):
        self.mental_predicate_set = mental_predicate_set
        self.action_predicate_set = action_predicate_set

        self.mental_time = []    # List[List[float]]
        self.mental_type = []    # List[List[List[int]]]
        self.action_time = []
        self.action_type = []

        for inst in data:
            # group mental events by (rounded) time to be robust to tiny float differences
            mental_dict = {}
            for elem in inst:
                t = round(float(elem['time_since_start']), 6)
                if elem['type_event'] in self.mental_predicate_set:
                    mental_dict.setdefault(t, []).append(elem['type_event'])

            times = sorted(mental_dict.keys())
            self.mental_time.append(times)
            self.mental_type.append([mental_dict[t] for t in times])

            # actions
            atimes, atypes = [], []
            for elem in inst:
                if elem['type_event'] in self.action_predicate_set:
                    atimes.append(elem['time_since_start'])
                    atypes.append(elem['type_event'])
            self.action_time.append(atimes)
            self.action_type.append(atypes)

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.mental_time[idx], self.mental_type[idx], \
               self.action_time[idx], self.action_type[idx]


# padding helpers used in collate_fn

def padding_batch(insts_time, insts_type):
    """Pad 2D lists to [B, T] arrays (times and types)."""
    max_len = max(len(inst) for inst in insts_time)
    batch_seq_time = np.array([
        inst + [PAD] * (max_len - len(inst))
        for inst in insts_time
    ], dtype=float)
    batch_seq_type = np.array([
        inst + [PAD] * (max_len - len(inst))
        for inst in insts_type
    ], dtype=int)
    return torch.tensor(batch_seq_time, dtype=torch.float32), torch.tensor(batch_seq_type, dtype=torch.long)


def pad_mask(padded_time, padded_type):
    return torch.where(padded_time == 0,
                       torch.zeros_like(padded_time),
                       torch.ones_like(padded_time))


def get_time_to_event(padded_time):
    diffs = torch.diff(padded_time, dim=-1)
    diffs = torch.where(diffs < 0, torch.zeros_like(diffs), diffs)
    return torch.nn.functional.pad(diffs, (1, 0), value=1e-3)


def collate_fn(insts):
    """Collate that returns mental placeholders and action time/type batch (we only need actions for replay)."""
    mental_time_list, mental_type_list, action_time_list, action_type_list = zip(*insts)
    action_time, action_type = padding_batch(action_time_list, action_type_list)
    pad_mask_action = pad_mask(action_time, action_type)
    action_time_to_event = get_time_to_event(action_time)
    # return mental info so upper code can optionally build mixed events
    return mental_time_list, mental_type_list, action_time, action_type, pad_mask_action, action_time_to_event
