import torchmetrics


class MetricsHelper:
    def __init__(
        self, metrics: str, stage: str, name: str, class_names, ignore_index=None
    ) -> None:
        self.stage = stage
        self.name = name
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.setup_metrcics(metrics)

    def setup_metrcics(self, metrics):
        self.macro_metrics = torchmetrics.MetricCollection(
            [
                metric(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    average="macro",
                )
                for metric in metrics
            ],
            prefix=f"{self.stage}/{self.name}/macro/",
        )
        self.nonaverage_metrics = torchmetrics.MetricCollection(
            [
                metric(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    average=None,
                )
                for metric in metrics
            ],
            prefix=f"{self.stage}/{self.name}/",
        )
        self.macro_metrics.to("cuda")
        self.nonaverage_metrics.to("cuda")
