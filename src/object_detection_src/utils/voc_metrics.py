import torch

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

'''
preds = [
 dict(
  boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0],[258.0, 41.0, 606.0, 285.0]]),
  scores=torch.tensor([0.536, 0.536]),
  labels=torch.tensor([0, 1]),
 )
]
'''

preds = [
 dict(
  boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],[214.0, 41.0, 562.0, 285.0]]),
  scores=torch.tensor([0.536, 0.536]),
  labels=torch.tensor([0, 1]),
 )
]

target = [
 dict(
  boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],[214.0, 41.0, 562.0, 285.0]]),
  labels=torch.tensor([0, 1]),
 )
]

metric = MeanAveragePrecision( class_metrics=True  )
metric.update(preds, target)

pprint(metric.compute())


def get_metric_value():
    return
