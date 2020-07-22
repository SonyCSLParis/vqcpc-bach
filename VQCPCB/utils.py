import torch
from torch import nn
import numpy as np


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return tensor.to('cuda')
    else:
        return tensor


def to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        if type(value) == list:
            print(f'{key.capitalize()}: [%s]' % ', '.join(map(str, value)))
        else:
            print(f'{key.capitalize()}: {value:.6}', end=endstr)


def categorical_crossentropy(value, target, mask=None):
    """

    :param value: list of (batch_size, num_events, num_tokens_of_corresponding_channel)
    :param target: (batch_size, num_events, num_channels)
    :param mask: (batch_size, num_events, num_channels)
    :return:
    """
    cross_entropy = nn.CrossEntropyLoss(size_average=False, reduce=False)
    sum = 0

    for channel_probs, target_channel, mask_channel in zip(value,
                                                           target.split(1, dim=2),
                                                           mask.split(1, dim=2)):
        # select relevent indices
        batch_size, num_events, num_tokens_of_channel = channel_probs.size()
        num_events_mask = mask_channel.sum() // batch_size
        assert mask_channel.sum() % batch_size == 0

        probs = channel_probs[mask_channel.bool().repeat(1, 1, num_tokens_of_channel)]
        target = target_channel[mask_channel.bool()]

        ce = cross_entropy(probs.view(-1, num_tokens_of_channel),
                           target.view(-1))
        sum = sum + ce
    return sum


def flatten(x):
    """

    :param x:(batch, num_events, num_channels, ...)
    :return: (batch, num_events * num_channels, ...) with num_channels varying faster
    """
    size = x.size()
    assert len(size) >= 3
    batch_size, num_events, num_channels = size[:3]
    remaining_dims = list(size[3:])
    x = x.view(batch_size, num_events * num_channels, *remaining_dims)
    return x


def unflatten(sequence, num_channels):
    """

    :param sequence: (batch_size, num_events * num_channels, ...)
    where num_channels is varying faster
    :return: (batch_size, num_events, num_channels, ...)
    """
    size = sequence.size()
    assert len(size) >= 2
    batch_size, sequence_length = size[:2]
    assert sequence_length % num_channels == 0
    num_events = sequence_length // num_channels
    remaining_dims = list(size[2:])

    chorale = sequence.view(batch_size, num_events, num_channels, *remaining_dims)
    return chorale


def timing_gpu():
    """
    Just to remember how to time gpus operation
    :return:
    """

    # notes ##################################
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'T: {elapsed_time_ms}')
    # notes ##################################


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def distilled_categorical_crossentropy(value, target, mask=None):
    """
    :param value: list of (batch_size, num_events, num_notes)
    :param target: list of (batch_size, num_events, num_notes)
    :return:
    """
    def cross_entropy_from_logits(p, q):
        """
        sum softmax(p) log softmax(q)
        :param p:
        :param q:
        :return:
        """
        p = torch.softmax(p, dim=1)
        log_term = q - torch.logsumexp(q, dim=1, keepdim=True)
        return -torch.sum(p * log_term, dim=1)

    sum = 0
    for channel, channel_target, channel_mask in zip(value, target, mask.split(1, dim=2)):
        # channel is (batch_size, num_events, num_tokens_of_corresponding_channel)
        # channel_target is (batch_size, num_events)
        for probs, label, m in zip(channel.split(1, dim=1),
                                   channel_target.split(1, dim=1),
                                   channel_mask.split(1, dim=1)):
            if m.squeeze(2).squeeze(1).float().mean().item() > 0.5:
                ce = cross_entropy_from_logits(label.squeeze(1),
                                               probs.squeeze(1))
                sum = sum + ce
    return sum
