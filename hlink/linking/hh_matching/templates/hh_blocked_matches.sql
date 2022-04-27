{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT
unmatched_a.*,
unmatched_b.*

FROM unmatched_a
JOIN unmatched_b
JOIN to_match

ON
unmatched_a.serialp_a == to_match.serialp_a
AND
unmatched_b.serialp_b == to_match.serialp_b
