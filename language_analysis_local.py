# copied and adapted from https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/language_analysis_local.py
# and https://github.com/jayelm/emergent-generalization/blob/master/code/emergence.py
# who both based on https://github.com/facebookresearch/EGG/blob/main/egg/core/language_analysis.py

import numpy as np
import torch
from egg.core.callbacks import Callback, ConsoleLogger
from egg.core.interaction import Interaction
import json
import editdistance
from scipy.spatial import distance
#from hausdorff import hausdorff_distance
from scipy.stats import spearmanr
from typing import Union, Callable
import pickle


class SavingConsoleLogger(ConsoleLogger):
    """Console logger that also stores the reported values"""
    def __init__(self, print_train_loss=False, as_json=False, n_metrics=2,
                 save_path: str = '', save_epoch: int = None):
        super(SavingConsoleLogger, self).__init__(print_train_loss, as_json)

        if len(save_path) > 0:
            self.save = True
            self.save_path = save_path
            self.save_epoch = save_epoch
            self.save_dict = {'loss_train': dict(),
                              'loss_test': dict()}
            for metric_idx in range(n_metrics):
                self.save_dict['metrics_train' + str(metric_idx)] = dict()
                self.save_dict['metrics_test' + str(metric_idx)] = dict()
        else:
            self.save = False

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

        if self.save:
            self.save_dict['loss_' + mode][epoch] = loss
            for idx, metric_key in enumerate(sorted(aggregated_metrics.keys())):
                self.save_dict['metrics_' + mode + str(idx)][epoch] = aggregated_metrics[metric_key]
            if epoch == self.save_epoch:
                with open(self.save_path + '/loss_and_metrics.pkl', 'wb') as handle:
                    pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        if self.save:
            with open(self.save_path + '/loss_and_metrics.pkl', 'wb') as handle:
                pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class MessageLengthHierarchical(Callback):
    """For every possible number of relevant attributes, take the messages for inputs with that number of relevant
    attributes and calculate the (absolute) difference between message length and number of relevant attributes."""

    def __init__(self, n_attributes, print_train: bool = True, print_test: bool = True, is_gumbel: bool = True,
                 save_path: str = '', save_epoch: int = None):

        self.print_train = print_train
        self.print_test = print_test
        self.is_gumbel = is_gumbel
        self.n_attributes = n_attributes

        if len(save_path) > 0 and save_epoch:
            self.save = True
            self.save_path = save_path
            self.save_epoch = save_epoch
            self.save_dict = {'message_length_train': dict(),
                              'message_length_test': dict()}
        else:
            self.save = False

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_difference_length_relevance(logs, 'train', epoch)

    def on_test_end(self, loss, logs, epoch):
        self.print_difference_length_relevance(logs, 'test', epoch)

    @staticmethod
    def compute_message_length(messages):
        max_len = messages.shape[1]
        #print(max_len)
        # replace all symbols with zeros from first zero element
        for m_idx, m in enumerate(messages):
            first_zero_index = torch.where(m == 0)[0][0]
            #print(torch.where(m==0))
            #print(first_zero_index)
            messages[m_idx, first_zero_index:] = torch.zeros((1, max_len - first_zero_index))
        # calculate message length
        print(messages)
        print(torch.sum(messages==0, dim=1))
        message_length = max_len - torch.sum(messages == 0, dim=1)
        return message_length

    @staticmethod
    def compute_message_length_hierarchical(messages, fixed_vectors):

        message_length = MessageLengthHierarchical.compute_message_length(messages)
        n_attributes = fixed_vectors.shape[1]
        number_same = torch.sum(fixed_vectors, dim=1)

        message_lengths = []
        for n in range(1, n_attributes + 1):
            hierarchical_length = message_length[number_same == n].float()
            message_lengths.append(hierarchical_length)
        message_length_step = [round(torch.mean(message_lengths[i]).item(), 3) for i in range(n_attributes)]

        return message_length_step

    def print_difference_length_relevance(self, logs: Interaction, tag: str, epoch: int):

        message = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        print("message", message)
        relevance_vector = logs.sender_input[:, -self.n_attributes:]
        print("relevance vector", relevance_vector)

        message_length_step = self.compute_message_length_hierarchical(message, relevance_vector)

        output = json.dumps(dict(message_length_hierarchical=message_length_step, mode=tag, epoch=epoch))
        print(output, flush=True)

        if self.save:
            self.save_dict['message_length_' + tag][epoch] = message_length_step
            if epoch == self.save_epoch:
                with open(self.save_path + '/message_length_hierarchical.pkl', 'wb') as handle:
                    pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        if self.save:
            with open(self.save_path + '/message_length_hierarchical.pkl', 'wb') as handle:
                pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def encode_input_for_topsim_hierarchical(sender_input, dimensions):
    n_features = np.sum(dimensions)
    n_attributes = len(dimensions)
    relevance_vectors = sender_input[:, -n_attributes:]
    sender_input_encoded = torch.zeros((len(sender_input), n_features + n_attributes))

    base_count = 0
    for i, dim in enumerate(dimensions):
        sender_input_encoded[relevance_vectors[:, i] == 0, base_count + i:base_count + i + dim] = (
            sender_input[relevance_vectors[:, i] == 0, base_count:base_count + dim])
        sender_input_encoded[relevance_vectors[:, i] == 1, base_count + i + dim] = 1
        base_count = base_count + dim

    return sender_input_encoded


def encode_target_concepts_for_topsim(sender_input):
    """
   NOTE: I should maybe encode fixed vectors like the relevance vectors in the hierarchical reference game
    """
    n_obs = sender_input.shape[0]
    n_objects = sender_input.shape[1]
    n_attributes = sender_input.shape[2]
    n_targets = int(n_objects/2)

    #print("sender_input", sender_input.shape)
    # select targets
    target_concepts = sender_input[:,:n_targets,:]
    #print("target concepts", target_concepts.shape)

    # maybe I should just calculate pairwise hausdorff distances here?

    #encoded_target_concepts = list(target_concepts)

    return target_concepts


def python_pdist(X, metric, **kwargs):
    """
    Function from https://github.com/jayelm/emergent-generalization/blob/master/code/emergence.py 
    who took it from https://github.com/scipy/scipy/blob/v1.6.0/scipy/spatial/distance.py#L2057-L2069
    Implements a workaround for scipy because scipy.distance.pdist() only takes 2d-arrays.
    """
    # number of observations
    m = len(X)
    #print("m", m)
    k = 0
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    #print("dm", dm.shape) # (365231,)
    #print("metric", metric)
    # go through all pairs of observations 
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            #print("Xi", X[i])
            #print("Xj", X[j])
            dm[k] = metric(X[i], X[j], **kwargs)[0] # index at zero because function returns more elements
            #print("hd", metric(X[i], X[j])) 
            k = k + 1
    return dm


class TopographicSimilarityConceptLevel(Callback):
    """
    Computes topographic similarity at the concept level.
    The hausdorff distance is used to compute the distance between two sets of objects, i.e. concepts.
    """

    def __init__(
            self,
            dimensions,
            sender_input_distance_fn: Union[str, Callable] = "directed_hausdorff",
            message_distance_fn: Union[str, Callable] = "edit",
            compute_topsim_train_set: bool = True,
            compute_topsim_test_set: bool = True,
            is_gumbel: bool = False,
            save_path: str = '',
            save_epoch: int = None,
    ):

        self.sender_input_distance_fn = sender_input_distance_fn
        self.message_distance_fn = message_distance_fn

        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set
        assert compute_topsim_train_set or compute_topsim_test_set

        self.is_gumbel = is_gumbel
        self.dimensions = dimensions

        if len(save_path) > 0 and save_epoch:
            self.save = True
            self.save_path = save_path
            self.save_epoch = save_epoch
            self.save_dict = {'topsim_train': dict(),
                              'topsim_test': dict()}
        else:
            self.save = False

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_train_set:
            self.print_message(logs, "train", epoch)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_test_set:
            self.print_message(logs, "test", epoch)

    @staticmethod
    def compute_topsim(
            meanings: torch.Tensor,
            messages: torch.Tensor,
            meaning_distance_fn: Union[str, Callable] = "hausdorff",
            message_distance_fn: Union[str, Callable] = "edit",
    ) -> float:
        """
        This function taken is from EGG
        https://github.com/facebookresearch/EGG/blob/ace483e30c99a5bc480d84141bcc6f4416e5ec2b/egg/core/language_analysis.py#L164-L199
        (but modified by Mu & Goodman (2021) to allow pure python pdist with lists when a distance fn is
        callable (rather than scipy coercing to 2d arrays))
        """

        distances = {
            "edit": lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
            "cosine": distance.cosine,
            "hamming": distance.hamming,
            "jaccard": distance.jaccard,
            "euclidean": distance.euclidean
        }

        slow_meaning_fn = True
        if meaning_distance_fn in distances:
            meaning_distance_fn_callable = distances[meaning_distance_fn]
            slow_meaning_fn = False
        elif meaning_distance_fn == "hausdorff":
            meaning_distance_fn_callable = distance.directed_hausdorff
        else:
            meaning_distance_fn_callable = meaning_distance_fn

        slow_message_fn = True
        if message_distance_fn in distances:
            message_distance_fn_callable = distances[message_distance_fn]
            slow_message_fn = False
        else:
            message_distance_fn_callable = message_distance_fn


        assert (
                meaning_distance_fn and message_distance_fn
        ), f"Cannot recognize {meaning_distance_fn} \
            or {message_distance_fn} distances"

        # compute distances between concepts
        if slow_meaning_fn:
            # Mu & Goodman implement a workaround because scipy pdist() only accepts 2-dimensional arrays
            meaning_dist = python_pdist(meanings, meaning_distance_fn_callable)
        else:
            meaning_dist = distance.pdist(meanings, meaning_distance_fn_callable)

        # compute distances between messages
        if slow_message_fn:
            message_dist = python_pdist(messages, message_distance_fn_callable)
        else:
            message_dist = distance.pdist(messages, message_distance_fn_callable)

        # topsim = negative spearman correlation of these two spaces
        topsim = spearmanr(meaning_dist, message_dist, nan_policy="raise").correlation

        return topsim

    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:

        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = messages[0:1000]
        messages = [msg.tolist() for msg in messages]
        sender_input = logs.sender_input[0:1000]

        # NOTE: encoding function from hierarchical reference game probably not needed if I can successfully use the hausdorff distance
        # for computing distances between concepts ("relevance" should be included implicitly)
        #encoded_sender_input = encode_input_for_topsim_hierarchical(sender_input, self.dimensions)
        encoded_target_concepts = encode_target_concepts_for_topsim(sender_input)
        topsim = self.compute_topsim(encoded_target_concepts, messages)
        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))

        print(output, flush=True)

        if self.save:
            self.save_dict['topsim_' + mode][epoch] = topsim
            if epoch == self.save_epoch:
                with open(self.save_path + '/topsim.pkl', 'wb') as handle:
                    pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        if self.save:
            with open(self.save_path + '/topsim.pkl', 'wb') as handle:
                pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)