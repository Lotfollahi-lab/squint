"""
Adversarial batch-invariance cross-entropy loss for VQNiche_Dual.

Reads pre-computed batch logits from the loss data dict (produced by
`BatchAdversaryHead` in the model's forward) and the per-cell batch
labels. Returns a standard CE loss multiplied by `wt_adv_batch`.

Note on the gradient direction: the classifier head puts the
GradientReversalLayer (GRL) on its input, so when this CE loss is
backpropagated:
  - the classifier's own params receive the *true* gradient (it learns
    to predict batch — it has to, otherwise it can't apply useful
    pressure on the encoder)
  - the encoder's params receive the *negated* gradient (it learns to
    make z_mlp un-predictable for the classifier — i.e. batch-invariant)
"""
import torch
import torch.nn.functional as F


def adversarial_batch_loss(
        batch_logits: torch.Tensor,
        batch_labels: torch.Tensor,
        wt_adv_batch: float = 1.0,
    ) -> torch.Tensor:
    """
    batch_logits: (B, n_batches)  — output of the BatchAdversaryHead
    batch_labels: (B,)            — per-cell batch IDs (long tensor)
    """
    return F.cross_entropy(batch_logits, batch_labels.long()) * wt_adv_batch
