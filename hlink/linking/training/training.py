# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask

from .link_step_ingest_file import LinkStepIngestFile
from .link_step_create_comparison_features import LinkStepCreateComparisonFeatures
from .link_step_train_and_save_model import LinkStepTrainAndSaveModel


class Training(LinkTask):
    def __init__(self, link_run):
        super().__init__(link_run)
        self.training_conf = "training"
        self.table_prefix = ""

    def get_steps(self):
        return [
            LinkStepIngestFile(self),
            LinkStepCreateComparisonFeatures(self),
            LinkStepTrainAndSaveModel(self),
        ]
