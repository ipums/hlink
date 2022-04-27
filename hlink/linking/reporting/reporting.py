# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from ..link_task import LinkTask

from .link_step_report_r2_percent_linked import LinkStepReportR2PercentLinked
from .link_step_report_representivity import LinkStepReportRepresentivity
from .link_step_export_crosswalk import LinkStepExportCrosswalk


class Reporting(LinkTask):
    def get_steps(self):
        return [
            LinkStepReportR2PercentLinked(self),
            LinkStepReportRepresentivity(self),
            LinkStepExportCrosswalk(self),
        ]
