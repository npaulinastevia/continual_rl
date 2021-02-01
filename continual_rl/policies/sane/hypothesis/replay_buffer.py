import torch
import torch.nn as nn
import numpy as np
import time
from continual_rl.utils.utils import Utils


class UnrecognizedTypeException(Exception):
    pass


class ReplayEntry(nn.Module):
    """
    Replay entries are owned per-hypothesis, as part of a replay buffer it serves to reinforce the boundary of that hypothesis.
    """
    def __init__(self, input_state, reward_received=None):
        super().__init__()
        # TODO: might be better to use *255 -> uint8. Inputs are usually uint8 by default anyway...
        self.input_state = input_state.detach().half()  # We don't cascade our changes up the pipeline. We make it half to minimize RAM usage
        self.reward_received = reward_received  # Populated after this entry "resolves" - gets discounted reward. (Except in clustering)
        self.action_log_prob = None
        self.selected_action = None

    def clone(self):
        # TODO: not currently cloning the random number? since it's random anyway...? TODO: is there a case where it should really be kept
        cloned_entry = ReplayEntry(self.input_state.clone())

        if self.reward_received is not None:
            cloned_entry.reward_received = self.reward_received.clone()

        if self.action_log_prob is not None:
            cloned_entry.action_log_prob = self.action_log_prob.clone()

        if self.selected_action is not None:
            cloned_entry.selected_action = self.selected_action.clone()

        return cloned_entry


class ReplayBufferFileBacked(object):
    def __init__(self, maxlen, observation_space, large_file_path):
        self.maxlen = maxlen
        self._buffers, _ = self._construct_buffers(maxlen, observation_space, large_file_path)

    def _construct_buffers(self, maxlen, observation_space, large_file_path):
        buffers = {}
        file_paths = []

        # Store observations
        observation_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                         shape=(maxlen, *observation_space.shape),
                                                                         dtype=observation_space.dtype.type)
        buffers["input_state"] = observation_buffers
        file_paths.append(file_path)

        # Store rewards received:
        reward_received_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                             shape=(maxlen, 1),
                                                                             dtype=torch.float32)
        buffers["reward_received"] = reward_received_buffers
        file_paths.append(file_path)

        # Store action_log_probs:
        action_log_prob_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                             shape=(maxlen, 1),
                                                                             dtype=torch.float32)
        buffers["action_log_prob"] = action_log_prob_buffers
        file_paths.append(file_path)

        # Store selected action:  (TODO: support non-discrete)
        selected_action_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                             shape=(maxlen, 1),
                                                                             dtype=torch.uint8)
        buffers["selected_action"] = selected_action_buffers
        file_paths.append(file_path)

        # Store whether this entry has been set:
        been_set_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                      shape=(maxlen, 1),
                                                                      dtype=torch.bool)
        buffers["been_set"] = been_set_buffers
        file_paths.append(file_path)

        return buffers, file_paths

    def _get_unset_indices(self):
        unset_indices = torch.where(~self._buffers["been_set"])[0]
        return unset_indices.tolist()

    def _add_from_replay_entry(self, entry):
        """
        Assumes the input is a list of ReplayEntries. Adds to the end if the buffer isn't fully populated, otherwise
        inserts into a random entry.
        """
        unset_indices = self._get_unset_indices()
        index_to_populate = unset_indices.pop(0) if len(unset_indices) > 0 else np.random.randint(0, self.maxlen)

        for key in self._buffers.keys():
            # been_set gets set separately
            if key == "been_set":
                continue

            # Remove the batch dimension before storage
            replay_value = entry.__dict__[key]
            self._buffers[key][index_to_populate].copy_(replay_value.squeeze(0))

        self._buffers["been_set"][index_to_populate] = True

    def _add_many_from_buffer_dict(self, buffer_dict):
        """
        Assumes the input is a dictionary much like the one stored in this class's buffer.
        """
        unset_indices = self._get_unset_indices()

        for buffer_dict_id in range(len(buffer_dict["been_set"])):
            index_to_populate = unset_indices.pop(0) if len(unset_indices) > 0 else np.random.randint(0, self.maxlen)

            for key in buffer_dict.keys():
                self._buffers[key][index_to_populate].copy_(buffer_dict[key][buffer_dict_id])

    def add_many(self, entries):
        # We can have lists of either ReplayEntrys or buffer-like dicts. This supports both
        if isinstance(entries, list):
            for entry in entries:
                self.add_many(entry)
        elif isinstance(entries, ReplayEntry):
            self._add_from_replay_entry(entries)
        elif isinstance(entries, dict):
            self._add_many_from_buffer_dict(entries)
        else:
            raise UnrecognizedTypeException(f"Entries are an unrecognized type: {type(entries)}")

    def get_random(self, num_to_get):
        max_index = len(self)
        indices = np.random.randint(0, max_index, size=num_to_get) if max_index > 0 else []
        buffer_subset = {}

        for key in self._buffers.keys():
            buffer_subset[key] = self._buffers[key][indices]

        return buffer_subset

    def get_all(self):
        """
        Gets all *filled* entries from the buffer.
        """
        indices = list(range(len(self)))
        buffer_subset = {}

        for key in self._buffers.keys():
            buffer_subset[key] = self._buffers[key][indices]

        return buffer_subset

    def __len__(self):
        unset_indices = self._get_unset_indices()
        return int(unset_indices[0] if len(unset_indices) > 0 else self.maxlen)


class ReplayBufferDataLoaderWrapper(object):
    """
    Wraps a replay buffer such that it can be used by a DataLoader
    """
    def __init__(self, replay_buffer):
        self._replay_buffer = replay_buffer

    def __len__(self):
        return len(self._replay_buffer)

    def __getitem__(self, item):
        assert self._replay_buffer._buffers["been_set"][item]
        input_state = self._replay_buffer._buffers["input_state"][item]
        reward_received = self._replay_buffer._buffers["reward_received"][item]
        action_log_prob = self._replay_buffer._buffers["action_log_prob"][item]
        selected_action = self._replay_buffer._buffers["selected_action"][item]
        return input_state, reward_received, action_log_prob, selected_action
