# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask
from .link_step_block_on_households import LinkStepBlockOnHouseholds
from .link_step_filter import LinkStepFilter
from hlink.linking.matching.link_step_score import LinkStepScore


class HHMatching(LinkTask):
    def get_steps(self):
        return [
            LinkStepBlockOnHouseholds(self),
            LinkStepFilter(self),
            LinkStepScore(self),
        ]

    def __init__(self, link_run):
        super().__init__(link_run, display_name="Household Matching")
        self.training_conf = "hh_training"
        self.table_prefix = "hh_"
