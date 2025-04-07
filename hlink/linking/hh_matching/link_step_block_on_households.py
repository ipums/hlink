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
        id_col = self.task.link_run.config["id_column"]
        records_to_match = self.task.link_run.config.get("hh_matching", {}).get(
            "records_to_match", "unmatched_only"
        )

        # Get the IDs for the potential matches that were deemed a match
        self.task.run_register_python(
            "indiv_matches",
            lambda: self.task.spark.table("predicted_matches")
            .select(f"{id_col}_a", f"{id_col}_b")
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
                pdfa, on=[individuals_matched[f"{id_col}_a"] == pdfa[f"{id_col}"]]
            )
            .select(individuals_matched[f"{id_col}_b"], pdfa.serialp.alias("serialp_a"))
            .join(pdfb, on=[individuals_matched[f"{id_col}_b"] == pdfb[f"{id_col}"]])
            .select("serialp_a", pdfb.serialp.alias("serialp_b"))
            .distinct()
        )

        self.task.run_register_python("serials_to_match", lambda: serials_to_match)

        if records_to_match == "unmatched_only":
            logger.debug("Excluding people who were already linked in step I")
            # Get the individual IDs and serialps of the people who were NOT matched in the first round
            self.task.run_register_python(
                "unmatched_a",
                lambda: pdfa.join(
                    individuals_matched,
                    on=[pdfa[f"{id_col}"] == individuals_matched[f"{id_col}_a"]],
                    how="left_anti",
                ).select(
                    pdfa[f"{id_col}"].alias(f"{id_col}_a"),
                    pdfa.serialp.alias("serialp_a"),
                ),
            )

            self.task.run_register_python(
                "unmatched_b",
                lambda: pdfb.join(
                    individuals_matched,
                    on=[pdfb[f"{id_col}"] == individuals_matched[f"{id_col}_b"]],
                    how="left_anti",
                ).select(
                    pdfb[f"{id_col}"].alias(f"{id_col}_b"),
                    pdfb.serialp.alias("serialp_b"),
                ),
            )
        elif records_to_match == "all":
            pdfa_renamed = pdfa.select(
                col(id_col).alias(f"{id_col}_a"), col("serialp").alias("serialp_a")
            )
            pdfb_renamed = pdfb.select(
                col(id_col).alias(f"{id_col}_b"), col("serialp").alias("serialp_b")
            )
            pdfa_renamed.write.saveAsTable("unmatched_a")
            pdfb_renamed.write.saveAsTable("unmatched_b")
        else:
            raise ValueError(
                f"Invalid choice for hh_matching.records_to_match: '{records_to_match}'"
            )

        uma = self.task.spark.table("unmatched_a")
        umb = self.task.spark.table("unmatched_b")
        stm = self.task.spark.table("serials_to_match")

        logger.debug(f"unmatched_a has {uma.count()} records")
        logger.debug(f"unmatched_b has {umb.count()} records")
        logger.debug(f"serials_to_match has {stm.count()} records")

        logger.debug("Blocking on household serial ID and generating potential matches")
        # Generate potential matches with those unmatched people who were in a household (serialp) with a match, blocking only on household id
        self.task.run_register_python(
            "hh_blocked_matches",
            lambda: stm.join(uma, "serialp_a").join(umb, "serialp_b").distinct(),
            persist=True,
        )

        hh_blocked_matches = self.task.spark.table("hh_blocked_matches")
        logger.debug(f"hh_blocked_matches has {hh_blocked_matches.count()} records")

        print(
            "Potential matches from households which contained a scored match have been saved to table 'hh_blocked_matches'."
        )
