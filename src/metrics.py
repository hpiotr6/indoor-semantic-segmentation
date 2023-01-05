import pandas as pd
import plotly.express as px
import torch
import torchmetrics
from torchmetrics import classification

import wandb


class WandbHelper:
    def __init__(self, stage: str, name: str, class_names, ignore_index=None) -> None:
        self.stage = stage
        self.name = name
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.setup_metrcics()

    def setup_metrcics(self):
        self.macro_metrics = torchmetrics.MetricCollection(
            {
                f"{self.name}_macro_iou": classification.MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    # validate_args=False,
                ),
                f"{self.name}_macro_acc": classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    # validate_args=False,
                ),
            },
            prefix=f"{self.stage}/",
        )

        self.nonaverage_metrics = torchmetrics.MetricCollection(
            {
                f"{self.name}_nonaverage_acc": classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average=None,
                    ignore_index=self.ignore_index,
                    # validate_args=False,
                ),
                f"{self.name}_nonaverage_iou": classification.MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    average=None,
                    ignore_index=self.ignore_index,
                    # validate_args=False,
                ),
                f"{self.name}_confusion_matrix": classification.MulticlassConfusionMatrix(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    normalize="true",
                    # validate_args=False,
                ),
            },
            prefix=f"{self.stage}/",
        )

        self.macro_metrics.to("cuda")
        self.nonaverage_metrics.to("cuda")

    def log_confusion_matrix(self, matrix: pd.DataFrame):

        fig = px.imshow(matrix, template="seaborn")
        wandb.log({f"{self.name} confusion Matrix": fig})

    def log_confusion_matrix_table(self, matrix: pd.DataFrame):
        # self.logger.log_table(key="conf", dataframe=matrix_df)
        tb = wandb.Table(data=matrix)
        wandb.log({f"{self.name} confusion_matrix": tb})

    def get_matrix_df(self, matrix: torch.Tensor):
        return pd.DataFrame(matrix.cpu(), self.class_names, self.class_names)

    # def log_polar(self, nonaverage_df: pd.DataFrame, r: str):

    #     fig = px.line_polar(
    #         nonaverage_df.iloc[1].T.reset_index(),
    #         r="test/MulticlassJaccardIndex",
    #         theta="index",
    #         line_close=True,
    #     )
    #     wandb.log({"Polar IOU": fig})
    #     self.logger.log_table(key="micro_metrics", dataframe=micro_metrics_df)

    def get_nonaverage_df(self, micro_metrics, metric_names):
        frame = torch.vstack(
            [micro_metrics[name] for name in metric_names],
        ).cpu()
        micro_metrics_df = pd.DataFrame(
            frame, index=metric_names, columns=self.class_names
        )

        return micro_metrics_df
