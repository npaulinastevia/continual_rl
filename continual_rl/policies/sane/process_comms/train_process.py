import torch
import time
from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayBuffer
from continual_rl.policies.sane.hypothesis_directory.utils import Utils


class TrainProcess(object):
    """
    The process that handles training hypothesis pattern filters. This process can handle multiple hypotheses.
    """
    def __init__(self, hypothesis_accessor, output_dir):
        self._hypothesis_accessor = hypothesis_accessor
        self._output_dir = output_dir
        self._hypotheses = {}  # Map from unique id to hypothesis

    def add_hypothesis(self, hypothesis):
        self._hypotheses[hypothesis.unique_id] = hypothesis

    def delete_hypothesis(self, hypothesis_id):
        if hypothesis_id in self._hypotheses:  # TODO: shouldn't be required, but deletions are getting called more than once and this is not top priority, but is making the real issue harder to find
            # For some reason the deletion is hanging randomly. Trying moving it to the CPU first.
            # Heisenbugs are the worst
            self._hypotheses[hypothesis_id].pattern_filter.cpu()
            del self._hypotheses[hypothesis_id]

    def process_queue(self, hypothesis_comms, train_process_logger):
        
        while True:
            next_message = hypothesis_comms.incoming_queue.get()

            if next_message is None:  # kill signal
                train_process_logger.info(f"Killing process")
                #Utils.delete_file_descriptors()  # TODO: necessary? Kind of a last-ditch - currently probably not helping
                break

            message_id, request_id, hypothesis_id, request_object, response_requested = next_message

            train_process_logger.info(f"Processing {message_id} for hypothesis_id {str(hypothesis_id)[:6]}")

            if message_id == "add_hypothesis":
                hypothesis = request_object
                self.add_hypothesis(hypothesis)
                request_result = {}

            elif message_id == "delete_hypothesis":
                hypothesis_id = request_object
                self.delete_hypothesis(hypothesis_id)
                request_result = {}

            else:
                # All messages below here require a specific hypothesis
                hypothesis = self._hypotheses[hypothesis_id]  # TODO: also get a hypothesis-specific logger? (Currently logger is proc-specific)

                if message_id == "train":
                    args = request_object[0]
                    kwargs = request_object[1]
                    self._hypothesis_accessor.try_train_pattern_filter(hypothesis, *args, **kwargs)
                    request_result = {}

                elif message_id == "clear_replay":
                    hypothesis._replay_buffer.clear()

                elif message_id == "add_many_to_replay":
                    entries = ReplayBuffer.inflate_from_bulk_transfer(request_object)
                    #cloned_buffer = [entry.clone().cpu() for entry in entries]  # TODO: this isn't enforced by anything...is it actually clearing memory on the original process? (TODO)
                    hypothesis._replay_buffer.add_many(entries)

                    hypothesis.replay_entries_since_last_train += len(entries)

                    request_result = {}

                elif message_id == "add_many_to_negative_examples":
                    entries = ReplayBuffer.inflate_from_bulk_transfer(request_object)
                    #cloned_buffer = [entry.clone().cpu() for entry in entries]  # TODO: this isn't enforced by anything...is it actually clearing memory on the original process? (TODO)
                    hypothesis._negative_examples.add_many(entries)

                    request_result = {}

                elif message_id == "get_replay_buffer_length":
                    request_result = len(hypothesis._replay_buffer)

                elif message_id == "get_random_replay_entries":
                    args = request_object[0]
                    kwargs = request_object[1]
                    entries = hypothesis._replay_buffer.get(*args, **kwargs)
                    #cloned_buffer = [entry.clone() for entry in entries]
                    bulk_tensor_obj = ReplayBuffer.prepare_for_bulk_transfer(entries)
                    request_result = bulk_tensor_obj

                elif message_id == "get_all_replay_entries":
                    #cloned_buffer = [entry.clone() for entry in hypothesis._replay_buffer]  # "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending"
                    bulk_tensor_obj = ReplayBuffer.prepare_for_bulk_transfer(hypothesis._replay_buffer)
                    request_result = bulk_tensor_obj

                elif message_id == "ping":
                    # TODO: this loading doesn't belong, being lazy to test it
                    self._hypothesis_accessor.load_learner(hypothesis)
                    if hypothesis._device.type == 'cuda':
                        train_process_logger.info("Synchronizing train_process hypothesis")
                        torch.cuda.synchronize(hypothesis._device)  # TODO: sometimes hangs?
                    request_result = {}

                else:
                    raise NotImplementedError(f"{next_message[0]} messages are not handled")

            del request_object  # Per memory-freeing best-practices here: https://pytorch.org/docs/stable/multiprocessing.html

            # Most requests return a result, so the caller knows it's done
            if response_requested:
                hypothesis_comms.outgoing_queue.put((request_id, request_result))

            train_process_logger.info(f"Finished processing {message_id} for hypothesis {hypothesis_id}")
            time.sleep(.1)  # TODO: ??? Shouldn't....be necessary? (Trying to debug intermittent hang)

    def try_process_queue(self, hypothesis_comms):
        train_process_logger = Utils.create_logger(f"{self._output_dir}/train_process_{hypothesis_comms.friendly_name}.log")
        
        try:
            self.process_queue(hypothesis_comms, train_process_logger)
        except Exception as e:
            train_process_logger.exception(f"Caught exception {e}. Shutting down the experiment")
            train_process_logger.info(f"Caught exception {e}. Shutting down the experiment")  # TODO: because not 100% sure I'm seeing the exception in my logs
            hypothesis_comms.outgoing_queue.put((None, None))
            time.sleep(0.5)  # Probably not necessary?

            raise e  # Re-raise to get the stack trace and such
