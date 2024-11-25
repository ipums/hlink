# Substitutions
* Parent header: `substitution_columns`
* Subheader name: `substitutions`
* Type: List
* Attributes:
  * `substitution_file` -- Type: `string`.  Required.  Path to the file containing the look-up table to join against for replacement values.

You must supply a substitution file and either specify `regex_word_replace=true` or supply a join value.

## 1:1 substitution by data table

Performs a 1:1 replacement on a filtered subset of the data table.  If the
input column data equals a value in the second column of the substitution file,
it is replaced with the data in the first column of the substitution file.
Used to replace variant name forms with standardized name forms, filtering on
a column like sex which may affect common names.

* Attributes:
  * `join_column` -- Type: `string`.  Column to filter input data on.
  * `join_value` -- Type: `string`.  Value to filter for in the input data.
  
```
[[substitution_columns]]
column_name = "namefrst_std"

[[substitution_columns.substitutions]]
join_column = "sex"
join_value = "1"
substitution_file = "/path/to/name_std/male.csv"

[[substitution_columns.substitutions]]
join_column = "sex"
join_value = "2"
substitution_file = "/path/to/name_std/female.csv"
```

## Substitution by regex word replace

Performs word replacement within a column's data string (such as replacing the abbreviation `Ave.` in the string `7th Ave.` with `Avenue` to create `7th Avenue`).

* Attributes:
  * `regex_word_replace` -- Type: `boolean`.  Whether or not to use regex matching on the input data to perform replacement.  If `true`, the swap value will still be replaced if it is anywhere in the column data, as long as it is:
    * at the start of the column data string, or proceeded by a space
    * at the end of the column data string, or followed by a space 

```
[[substitution_columns]]
column_name = "street_unstd"

[[substitution_columns.substitutions]]
regex_word_replace = true
substitution_file = "/path/to/dir/substitutions_street_abbrevs.csv"
```
