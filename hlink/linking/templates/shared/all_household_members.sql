{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT 
{% if selects_a is defined %}
  {{ ",\n".join(selects_a) }}
{% else %}
  {% include 'shared/includes/all_household_members_selects_a.sql' %}
{% endif %}
FROM raw_df_a ra
JOIN {{ hh_keeps if hh_keeps is defined else "hh_keeps" }} hc ON ra.NEW_SERIAL_a = hc.NEW_SERIAL_a
LEFT JOIN {{ plinks if plinks is defined else "plinks_round_1_2_3" }} pl ON pl.id_a = ra.id AND pl.SERIAL_b = hc.SERIAL_b

UNION ALL

SELECT 
{% if selects_b is defined %}
  {{ ",\n".join(selects_b) }}
{% else %}
  {% include 'shared/includes/all_household_members_selects_b.sql' %}
{% endif %}
FROM raw_df_b rb
JOIN {{ hh_keeps if hh_keeps is defined else "hh_keeps" }} hc ON rb.SERIAL = hc.SERIAL_b
LEFT JOIN {{ plinks if plinks is defined else "plinks_round_1_2_3" }} pl ON pl.id_b = rb.id AND pl.NEW_SERIAL_a = hc.NEW_SERIAL_a
WHERE pl.SERIAL_a IS NULL
{% if order_bys %}
  ORDER BY {{ ", ".join(order_bys) }}
{% else %}
	ORDER BY NEW_SERIAL_a, SERIAL_b, NAMEFRST_MAX_JW DESC
{% endif %}
