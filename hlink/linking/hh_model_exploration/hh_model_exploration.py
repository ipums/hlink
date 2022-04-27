# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask

from hlink.linking.model_exploration.link_step_ingest_file import LinkStepIngestFile
from hlink.linking.model_exploration.link_step_create_features import (
    LinkStepCreateFeatures,
)
from hlink.linking.model_exploration.link_step_train_test_models import (
    LinkStepTrainTestModels,
)


class HHModelExploration(LinkTask):
    def __init__(self, link_run):
        super().__init__(
            link_run,
            display_name="Household Model Exploration",
        )
        self.training_conf = "hh_training"
        self.table_prefix = "hh_model_eval_"

    def get_steps(self):
        return [
            LinkStepIngestFile(self),
            LinkStepCreateFeatures(self),
            LinkStepTrainTestModels(self),
        ]
