import random
import torch


###################################
# Embedding learning losses
def nce_loss(positive, negatives, temperature):
    """

    :param positive: b * k
    :param negatives: b * k * num_negatives
    :return:
    """

    positive = positive / temperature
    negatives = negatives / temperature

    negatives_and_positive = torch.cat(
        [
            negatives,
            positive.unsqueeze(2)
        ], dim=2
    )
    normalizer = torch.logsumexp(negatives_and_positive, dim=2)

    loss_batch = positive - normalizer

    # sum over k, mean over batches
    loss = -loss_batch.sum(1).mean(0)
    # loss = -torch.mean(loss_batch)
    return loss


def nce_loss_entmax(positive, negatives, temperature):
    """

    :param positive: b * k
    :param negatives: b * k * num_negatives
    :return:
    """
    negatives_and_positive = torch.cat(
        [
            positive.unsqueeze(2),
            negatives
        ], dim=2
    )
    from entmax import entmax15
    entmax = entmax15(negatives_and_positive, dim=2)

    loss_batch = torch.log(entmax[:, :, 0] + 1e-8)

    # sum over k, mean over batches
    loss = -loss_batch.sum(1).mean(0)
    # loss = -torch.mean(loss_batch)
    return loss


def triplet_loss(z_a, z_p, z_n):
    """

    :param positive: b * k
    :param negatives: b * k * num_negatives
    :return:
    """
    alpha = 0.1

    batch_size, num_negative_samples, _, _ = z_n.shape
    z_a_negative = z_a.unsqueeze(1).repeat(1, num_negative_samples, 1, 1)

    # Compute distances
    D_ap = torch.norm((z_a - z_p), p=2, dim=2)
    D_an_all = torch.norm((z_a_negative - z_n), p=2, dim=3)

    # select negative samples (hard, semi-hard, distance sampling, random)
    #  random for now
    D_an = torch.zeros_like(D_ap)
    for batch_ind in range(batch_size):
        selected_negative_ind = random.randint(0, num_negative_samples - 1)
        D_an[batch_ind] = D_an_all[batch_ind, selected_negative_ind]

    # loss
    contrast = D_ap - D_an + alpha
    loss_batch = torch.where(contrast > 0, contrast, torch.zeros_like(contrast))
    loss = loss_batch.sum(1).mean(0)
    return loss


def contrastive_loss(positive, negatives):
    """

    :param positive: b * k
    :param negatives: b * k * num_negatives
    :return:
    """
    # TODO
    raise NotImplementedError


###################################
#  Regularisation
def attention_entropy_loss(attentions_decoder, attention_encoder):
    """

    :param attentions: dictionnary containing the different attentions from different transformers and layers
    :return:
    """

    def this_attn_entropy(v):
        this_entropy = - (v * torch.log(v + 1e-20)).sum(-1)
        #  sum along heads, mean along batch and input dim
        this_entropy = this_entropy.mean()
        return this_entropy

    attention_entropy = []
    for attn_decoder, attn_encoder in zip(attentions_decoder, attention_encoder):
        for _, v in attn_decoder.items():
            attention_entropy.append(this_attn_entropy(v))
        for _, v in attn_encoder.items():
            attention_entropy.append(this_attn_entropy(v))
    return attention_entropy
