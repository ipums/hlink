# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask

from .link_step_register_raw_dfs import LinkStepRegisterRawDfs
from .link_step_prep_dataframes import LinkStepPrepDataframes


class Preprocessing(LinkTask):
    def get_steps(self):
        return [LinkStepRegisterRawDfs(self), LinkStepPrepDataframes(self)]
