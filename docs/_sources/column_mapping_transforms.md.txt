# Column mapping transforms

Each header below represents a column mapping transform type.  Transforms are used in the context of `column_mappings`.

Some transforms refer to "a" or "b". These mean the transforms apply to columns from only one of the two datasets to be linked (we're trying to link people in dataset "a" with people in dataset "b").

More than one transform can be applied to a column. Transforms apply in the order they're listed, so the output of one transform may be the input of another.

Each transform applies to the column specified by the `column_name` attribute in the config under the `[[column_mappings]]` section. The `transforms` attribute
indicates the type of the transform, which is one of the ones listed below. Along with `type`, there can be additional attributes used by the transform.
These may vary by type, and additional information is given for each type of transform below. Often an additional attribute is just named `value` or `values`.

```
[[column_mappings]]
alias = "namefrst_split"
column_name = "namefrst_clean"
transforms = [ { type = "split" } ]
```

## add_to_a

Add a value to a column from dataset "a".

```
transforms = [ { type = "add_to_a", value = 11 } ]
```

## concat_to_a

Concatenate the string value to the end of a column in dataset "a".

```
transforms = [ { type = "concat_to_a", value = " "} ]
```

## concat_to_b

Concatenate the string value to the end of a column in dataset "b".

```
transforms = [ { type = "concat_to_b", value = " "} ]
```

## lowercase_strip

Used in name cleaning. 

Convert alphabetical characters to lower-case and strip white space characters from the start and end of the strings in the column.

```
transforms = [ { type = "lowercase_strip"} ]

```

## rationalize_name_words

Used in name cleaning.

Replace '?', '\*', and '-' with spaces. Since people's names in raw census data can contain these
characters, replacing these characters can lead to better matching.

```
transforms = [ { type = "rationalize_name_words"} ]
```


## remove_qmark_hyphen

Used in name cleaning.

Remove the '?-' from words and replace with nothing.

```
transforms = [ { type = "remove_qmark_hyphen"} ]
```

## remove_punctuation

Remove most punctuation and replace with nothing. 

Removes:
```
? - \ / " ' : , . [ ] { }
```

```
transforms = [ { type = "remove_punctuation"} ]
```

## replace_apostrophe

Used in name cleaning.

Replace each apostrophe "'" with a space.

```
transforms = [ { type = "replace_apostrophe"} ]

```


##  remove_alternate_names

Used in name cleaning.

Remove any names following the string 'or'.

```
transforms = [ { type = "remove_alternate_names"} ]
```


## remove_suffixes

Used in name cleaning.

Given a list of suffixes, remove them from the names in the column.

```
transforms=[{ type = "remove_suffixes",  values = ["jr", "sr", "ii", "iii"] }]
```

## remove_stop_words

Used in name cleaning.

Remove last words from names such as street names.

```
transforms=[
{type = "remove_stop_words", values = ['alley','ally','aly','anex','annex','av','ave','aven','avenu','avenue','avn','avnue','avanue','avaneu','bg','blvd','boul','boulevard','brg','bridge','burg','camp','circle','cor', 'corner', 'corners','cors', 'court', 'courts', 'cp', 'cres', 'crescent', 'ct', 'cts', 'dr','driv', 'drive', 'est', 'estate', 'express', 'expressway', 'ext', 'extension', 'ferry', 'fort', 'frt', 'fry', 'ft', 'heights', 'ht', 'hts', 'is', 'island', 'key', 'ky', 'ldg', 'lodge', 'mill', 'mills', 'ml', 'mls', 'mount', 'mountain', 'mountin', 'mt', 'mtn', 'park', 'parkway','pike', 'pikes','pkwy', 'pl', 'place', 'point', 'points', 'pr', 'prairie', 'prk', 'pt', 'pts', 'rad', 'radial', 'rd', 'rds', 'rest', 'riv', 'river', 'road', 'roads', 'rst', 'spgs', 'springs', 'sq', 'square', 'st', 'sta', 'station', 'str', 'street', 'streets', 'strt', 'sts', 'ter', 'terrace', 'track', 'tracks', 'trail', 'trails', 'trnpk', 'turnpike', 'un', 'union', 'valley', 'vally', 'via', 'viaduct', 'vill', 'villag', 'village', 'villiage', 'well', 'wl', 'wl', 'and','of','.',',','-','/','&','south','north','east','west','s','n','e','w','block']}]
  
```

## remove_prefixes

Used in name cleaning.

Remove prefixes like "Ms.", "Mr.", or "Mrs." from names.

In some census data, "ah" is such a prefix from Chinese names.

```
transforms=[{ type = "remove_prefixes", values = ["ah"]}]
```

## condense_strip_whitespace

Used in name cleaning.

Take white space that may be more than one character or contain non-space characters and replace it with a single space.

```

transforms=[{ type = "condense_strip_whitespace"}]

```

## remove_one_letter_names

Used in name cleaning.

If a name is a single character, remove it and leave the white space behind.

```
transforms=[{ type = "remove_one_letter_names"}]
```


## split


Split the column value on space characters (" ").

```
[[column_mappings]]
alias = "namefrst_split"
column_name = "namefrst_clean"
transforms = [ { type = "split" } ]
```
 
 
 

## array_index

If the column contains an array, select the element at the given position.

This can be used as the input to another transform. In the example below, the first transform selects the second (index 1) item from  the "namefrst_split" column that contains a set of names split on white space. Then, the substring 0,1 is selected which gives the first initial of the person's probable middle name.

```
alias = "namefrst_mid_init"
column_name = "namefrst_split"
transforms = [
 { type = "array_index", value = 1},
 { type = "substring", values = [0, 1]}
]
```

## mapping

Map single or multiple values to a single output value, otherwise known as a "recoding."

```
[[column_mappings]]
column_name = "birthyr"
alias = "clean_birthyr"
transforms=[
{type = "mapping"
values = [{"from"=[9999,1999], "to" = ""},
{"from" = -9998, "to" = 9999}
]}
```

## substring

Replace a column with a substring of the data in the column.

```
transforms = [
 { type = "substring", values = [0, 1]}]
 ```

## divide_by_int

Divide data in a column by an integer value. It may leave a non-integer result. 

For instance, this transform takes the birthplace variable and converts it from the detailed version to the general version. The two least significant digits are detailed birthplace information; to make the more general version, we simply drop them by dividing by 100 and rounding to the lowest whole number (floor function).

```
[[column_mappings]]
column_name = "bpl"
alias = "bpl_root"
transforms = [
  { type = "divide_by_int", value = 100 },
  { type = "get_floor" }
]

```


## when_value


Apply conditional logic to replacement of values in a column. Works like the SQL if() or case() expressions in the SQL "select" clause.

When a the value of a column is "value" replace it with "if_value" otherwise replace it with the "else_value".

This example replaces all "race" IPUMS codes with 0 (white) or 1 (non-white). An IPUMS code of 100 is the "white" race category.

```
column_name = "race"
transforms = [
  { type = "when_value", value = 100, if_value = 0, else_value = 1}
]
```


## get_floor

Round down to the nearest whole number. 

This example produces the general version of the IPUMS "relate" variable. The variable is coded such that detailed categories are between the hundreds (300 is child of household head, 301 is simply 'child', 302 is adopted child, 303 is step-child for instance). The general categories are usually all that's needed (1 == household head, 2 == spouse, 3 == child, 4 == child-in-law, 5 == parent, 6 == parent-in-law, 7== sibling, 12 == not related to head).

```
[[column_mappings]]
alias = "relate_div_100"
column_name = "relate"
transforms = [
  { type = "divide_by_int", value = 100 },
  { type = "get_floor" }
]
```

