# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask

from .link_step_explode import LinkStepExplode
from .link_step_match import LinkStepMatch
from .link_step_score import LinkStepScore


class Matching(LinkTask):
    def __init__(self, link_run):
        super().__init__(link_run)
        self.training_conf = "training"
        self.table_prefix = ""

    def get_steps(self):
        return [LinkStepExplode(self), LinkStepMatch(self), LinkStepScore(self)]
