import torch


###################################
# Embedding learning losses
def nce_loss(positive, negatives):
    """

    :param positive: b * k
    :param negatives: b * k * num_negatives
    :return:
    """

    positive = positive
    negatives = negatives

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
