# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging

from pyspark.sql.functions import col

from hlink.linking.link_step import LinkStep


logger = logging.getLogger(__name__)


class LinkStepBlockOnHouseholds(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "block on households",
            input_table_names=["predicted_matches", "prepped_df_a", "prepped_df_b"],
            output_table_names=["hh_blocked_matches"],
        )

    def _run(self):
        config = self.task.link_run.config
        id_col = config["id_column"]
        id_a = f"{id_col}_a"
        id_b = f"{id_col}_b"
        hhid_col = "serialp"
        hhid_a = f"{hhid_col}_a"
        hhid_b = f"{hhid_col}_b"

        logging.debug(f"The id column is {id_col}")
        logging.debug(f"The household ID column is {hhid_col}")

        records_to_match = config.get("hh_matching", {}).get(
            "records_to_match", "unmatched_only"
        )
        logger.debug(f"hh_matching.records_to_match is {records_to_match}")

        # Get the IDs for the potential matches that were deemed a match
        self.task.run_register_python(
            "indiv_matches",
            lambda: self.task.spark.table("predicted_matches")
            .select(id_a, id_b)
            .distinct(),
            persist=True,
        )

        pdfa = self.task.spark.table("prepped_df_a")
        pdfb = self.task.spark.table("prepped_df_b")
        individuals_matched = self.task.spark.table("indiv_matches")

        logger.debug(f"prepped_df_a has {pdfa.count()} records")
        logger.debug(f"prepped_df_b has {pdfb.count()} records")
        logger.debug(f"indiv_matches has {individuals_matched.count()} records")

        logger.debug("Getting household serial IDs for matched individuals")
        # Get the HH serial ids for these matched individuals
        serials_to_match = (
            individuals_matched.join(
                pdfa, on=[individuals_matched[id_a] == pdfa[id_col]]
            )
            .select(individuals_matched[id_b], pdfa[hhid_col].alias(hhid_a))
            .join(pdfb, on=[individuals_matched[id_b] == pdfb[id_col]])
            .select(hhid_a, pdfb[hhid_col].alias(hhid_b))
            .distinct()
        )

        self.task.run_register_python("serials_to_match", lambda: serials_to_match)

        if records_to_match == "unmatched_only":
            logger.debug("Excluding people who were already linked in step I")
            # Get the individual IDs and household IDs of the people who were NOT matched in the first round
            unmatched_a = pdfa.join(
                individuals_matched,
                on=[pdfa[id_col] == individuals_matched[id_a]],
                how="left_anti",
            )

            unmatched_b = pdfb.join(
                individuals_matched,
                on=[pdfb[id_col] == individuals_matched[id_b]],
                how="left_anti",
            )
        elif records_to_match == "all":
            logger.debug("Including people who were already linked in step I")
            unmatched_a = pdfa
            unmatched_b = pdfb
        else:
            raise ValueError(
                f"Invalid choice for hh_matching.records_to_match: '{records_to_match}'"
            )

        unmatched_a_selected = unmatched_a.select(
            col(id_col).alias(id_a), col(hhid_col).alias(hhid_a)
        )
        unmatched_b_selected = unmatched_b.select(
            col(id_col).alias(id_b), col(hhid_col).alias(hhid_b)
        )
        self.task.run_register_python("unmatched_a", lambda: unmatched_a_selected)
        self.task.run_register_python("unmatched_b", lambda: unmatched_b_selected)

        uma = self.task.spark.table("unmatched_a")
        umb = self.task.spark.table("unmatched_b")
        stm = self.task.spark.table("serials_to_match")

        logger.debug(f"unmatched_a has {uma.count()} records")
        logger.debug(f"unmatched_b has {umb.count()} records")
        logger.debug(f"serials_to_match has {stm.count()} records")

        logger.debug("Blocking on household serial ID and generating potential matches")
        # Generate potential matches with those unmatched people who were in a household with a match, blocking only on household id
        self.task.run_register_python(
            "hh_blocked_matches",
            lambda: stm.join(uma, hhid_a).join(umb, hhid_b).distinct(),
            persist=True,
        )

        hh_blocked_matches = self.task.spark.table("hh_blocked_matches")
        logger.debug(f"hh_blocked_matches has {hh_blocked_matches.count()} records")

        print(
            "Potential matches from households which contained a scored match have been saved to table 'hh_blocked_matches'."
        )
