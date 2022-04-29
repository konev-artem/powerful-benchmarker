from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer



class BarlowTwinLossAndTripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """
    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        #my_changes 128 is embeding size
        #self.bn = torch.nn.BatchNorm1d(128, affine=False)
        self.bn = None

    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def compute_loss(self, embeddings, labels, indices_tuple):

        if self.bn is None:
          self.bn = torch.nn.BatchNorm1d(embeddings.shape[-1], affine=False, device = embeddings.get_device())

        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(an_dists, ap_dists)
        if self.smooth_loss:
            loss = torch.log(1 + torch.exp(-current_margins))
        else:
            loss = torch.nn.functional.relu(-current_margins + self.margin)

        z1 = embeddings[::2]
        z2 = embeddings[1:][::2]

        # empirical cross-correlation matrix

        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        #my_changes
        #c.div_(self.args.batch_size)
        c.div_(256)
        #my_changes
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        #my_changes
        #off_diag = self.off_diagonal(c).pow_(2).sum()
        n, m = c.shape
        assert n == m
        off_diag = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        off_diag = off_diag.pow_(2).sum()

        #my_changes
        lambd = 1e-1

        loss_barlow_twins = on_diag + lambd * off_diag
        return {"loss": {"losses": 0*loss + loss_barlow_twins, "indices": indices_tuple, "reduction_type": "triplet"}}

    def get_default_reducer(self):
        return AvgNonZeroReducer()
