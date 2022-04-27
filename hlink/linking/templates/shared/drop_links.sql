{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT starting.*
FROM {{ starting_links }} starting
{% for links_to_drop in links_to_drop_list %}
LEFT JOIN {{ links_to_drop }} dropping_a_{{ loop.index }} ON dropping_a_{{ loop.index }}.id_a = starting.id_a AND dropping_a_{{ loop.index }}.NEW_SERIAL_a = starting.NEW_SERIAL_a AND dropping_a_{{ loop.index }}.SERIAL_b = starting.SERIAL_b
LEFT JOIN {{ links_to_drop }} dropping_b_{{ loop.index }} ON dropping_b_{{ loop.index }}.id_b = starting.id_b AND dropping_b_{{ loop.index }}.NEW_SERIAL_a = starting.NEW_SERIAL_a AND dropping_b_{{ loop.index }}.SERIAL_b = starting.SERIAL_b
{% endfor %}
WHERE
{% for links_to_drop in links_to_drop_list %}
{{ 'AND' if not loop.first else '' }} dropping_a_{{ loop.index }}.PERNUM_a IS NULL AND dropping_b_{{ loop.index }}.PERNUM_b IS NULL
{% endfor %}
