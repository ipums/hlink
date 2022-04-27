# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask

from hlink.linking.training.link_step_ingest_file import LinkStepIngestFile
from hlink.linking.training.link_step_create_comparison_features import (
    LinkStepCreateComparisonFeatures,
)
from hlink.linking.training.link_step_train_and_save_model import (
    LinkStepTrainAndSaveModel,
)


class HHTraining(LinkTask):
    def __init__(self, link_run):
        super().__init__(link_run, display_name="Household Training")
        self.training_conf = "hh_training"
        self.table_prefix = "hh_"

    def get_steps(self):
        return [
            LinkStepIngestFile(self),
            LinkStepCreateComparisonFeatures(self),
            LinkStepTrainAndSaveModel(self),
        ]
