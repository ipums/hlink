# Column Mappings

## Basic Usage

Each column mapping reads a column from the input datasets into hlink. It has a
`column_name` attribute which specifies the name of the input column to read in
from both datasets. Optionally, it may have an `alias` attribute which gives a
new name to use for the column in hlink.

Column mappings support some *transforms* which make changes to the data as they
are read in. These changes support data cleaning and harmonization. The available
column mapping transforms are listed below in the [transforms](#transforms) section.

## Advanced Usage

By default, the input column must have the same name in both input datasets.
With the `override_column_a` and `override_column_b` attributes, you can
specify a different name for either dataset A or dataset B. When you do this,
the `transforms` attribute applies only to the non-override dataset. You can also
provide an `override_transforms` attribute which applies only to the override
dataset.

## Transforms

Each section below describes a column mapping transform type. Each transform
operates on a single input column and outputs a single output column. More than
one transform may be applied to a column. Transforms apply in the order that
they are listed in the `transforms` list, so the output of one transform may
be the input of another. Input and output column types are listed in the format
"Maps input column type → output column type". The letters T and U represent
arbitrary column types.

Each transform requires a `type` attribute, which must be one of the names
listed below. Some transforms may use additional attributes. These vary by
type, and additional information appears for each type of transform in its
section below.

Some transforms are suffixed by "a" or "b". These suffixes mean that the
transforms apply to columns from only one of the two datasets to be linked
(dataset A or dataset B). Most transforms operate on both dataset A and dataset
B independently.

For example, if you have two datasets taken 10 years apart, you may want to
standardize the `age` variable so that it is comparable between the two
datasets. To do this, you could create a new `age_at_dataset_b` variable by
reading in the `age` variable from each dataset and then adding 10 to the
variable from dataset A with the `add_to_a` transform.

```
[[column_mappings]]
alias = "age_at_dataset_b"
column_name = "age"
transforms = [
    {type = "add_to_a", value = 10}
]
```

As another example, suppose that both datasets record each person's first name
as a string. In dataset A the variable is called `namefrst` and is entirely
lowercase, but in dataset B it is called `first_name` and is entirely uppercase.
You could read these two columns into a `namefrst` column in hlink and apply
a lowercase transform to only dataset B with the following configuration section.

```
[[column_mappings]]
alias = "namefrst"
column_name = "namefrst"
# Read from column first_name in dataset B
override_column_b = "first_name"
# Apply these transforms only to dataset B
override_transforms = [
    {type = "lowercase_strip"}
]
```



### add_to_a

Add the given `value` to a column from dataset A.

Maps numerical → numerical.

```
transforms = [{type = "add_to_a", value = 11}]
```

### concat_to_a

Concatenate the string `value` to the end of a column in dataset A.

Maps string → string.

```
transforms = [{type = "concat_to_a", value = " "}]
```


### concat_to_b

Concatenate the string `value` to the end of a column in dataset B.

Maps string → string.

```
transforms = [{type = "concat_to_b", value = " "}]
```

### concat_two_cols

Concatenate the values from two columns together as strings. This transform takes
a `column_to_append` attribute, which specifies the name of the column to concatenate
to the end of the mapped column. To concatenate more than two columns, you can
use this transform multiple times in a row.

If either of the columns are numerical, they are automatically converted to strings
before the concatenation.

Maps (string | numerical) → string.

```toml
# Concatenate two columns to the end of the mapped column.
transforms = [
    {type = "concat_two_cols", column_to_append = "statefip"},
    {type = "concat_two_cols", column_to_append = "county"},
]
```

### lowercase_strip

Used in name cleaning. Convert alphabetical characters to lower-case and strip white
space characters from the start and end of the strings in the column.

Maps string → string.

```
transforms = [{type = "lowercase_strip"}]
```

### rationalize_name_words

Used in name cleaning. Replace the characters `?`, `*`, and `-` with spaces. Since
people's names in raw census data can contain these characters, replacing these characters
can lead to better matching.

Maps string → string.

```
transforms = [{type = "rationalize_name_words"}]
```


### remove_qmark_hyphen

Used in name cleaning. Remove the characters `?` and `-` from strings in the column.

Maps string → string.

```
transforms = [{type = "remove_qmark_hyphen"}]
```

### remove_punctuation

Remove most punctuation from strings in the column. This transform removes these characters:
`? - \ / " ' : , . [ ] { }`.

Maps string → string.

```
transforms = [{type = "remove_punctuation"}]
```

### replace_apostrophe

Used in name cleaning. Replace each apostrophe `'` with a space.

Maps string → string.

```
transforms = [{type = "replace_apostrophe"}]

```

### remove_alternate_names

Used in name cleaning. If a string in the column contains the string ` or ` ("or" surrounded by spaces),
then remove the ` or ` and all following characters.

Maps string → string.

```
transforms = [{type = "remove_alternate_names"}]
```

### remove_suffixes

Used in name cleaning. Given a list of suffixes, remove them from the strings in the column.

Maps string → string.

```
transforms = [
    {
        type = "remove_suffixes",
        values = ["jr", "sr", "ii", "iii"]
    }
]
```

### remove_stop_words

Used in name cleaning. Remove last words from names such as street names.

Maps string → string.

```
transforms = [
    {
        type = "remove_stop_words",
        values = ['avenue', 'blvd', 'circle', 'court', 'road', 'street']
    }
]
```

### remove_prefixes

Used in name cleaning. Remove prefixes like "Ms.", "Mr.", or "Mrs." from names.

Maps string → string.

```
# In some census data, "ah" is a prefix from Chinese names.
transforms = [{type = "remove_prefixes", values = ["ah"]}]
```

### condense_strip_whitespace

Used in name cleaning. Take white space that may be more than one character or contain
non-space characters and replace it with a single space.

Maps string → string.

```
transforms = [{type = "condense_strip_whitespace"}]
```

### remove_one_letter_names

Used in name cleaning. If a name is a single character, remove it and leave the white space behind.

Maps string → string.

```
transforms = [{type = "remove_one_letter_names"}]
```

### split

Split the column value on space characters.

Maps string → array of string.

```
[[column_mappings]]
alias = "namefrst_split"
column_name = "namefrst_clean"
transforms = [{type = "split"}]
```

### array_index

If the column contains an array, select the element at the given position.

This can be used as the input to another transform. In the example below, the first transform selects the second (index 1) item from  the "namefrst_split" column that contains a set of names split on white space. Then the substring 0,1 is selected, which gives the first initial of the person's probable middle name.

Maps array of T → T.

```
[[column_mappings]]
alias = "namefrst_mid_init"
column_name = "namefrst_split"
transforms = [
    {type = "array_index", value = 1},
    {type = "substring", values = [0, 1]}
]
```

### mapping

Explicitly map from input values to output values. This is also known as a "recoding".
Input values which do not appear in the mapping are unchanged. By default, the output
column is of type string, but you can set `output_type = "int"` to cast the output
column to type integer instead.

Maps T → U.

```toml
[[column_mappings]]
column_name = "birthyr"
alias = "clean_birthyr"

[[column_mappings.transforms]]
type = "mapping"
mappings = {9999 = "", 1999 = "", "-9998" = "9999"}
output_type = "int"
```

*Changed in version 4.0.0: The deprecated `values` key is no longer supported.
Please use the `mappings` key documented above instead.*

### substring

Replace a column with a substring of the data in the column.

Maps string → string.

```
transforms = [
    {type = "substring", values = [0, 1]}
]
 ```

### divide_by_int

Divide data in a column by an integer value. It may leave a non-integer result.

For instance, the following example takes the birthplace variable and converts it
from the detailed version to the general version. The two least significant digits
are detailed birthplace information; to make the more general version, we simply drop
them by dividing by 100 and rounding to the lowest whole number (floor function).

Maps numerical → numerical.

```
[[column_mappings]]
column_name = "bpl"
alias = "bpl_root"
transforms = [
    {type = "divide_by_int", value = 100},
    {type = "get_floor"}
]
```


### when_value

Apply conditional logic to replacement of values in a column. Works like the SQL `if()` or `case()` expressions in the SQL `select` clause.
When the value of a column is `value` replace it with `if_value`. Otherwise replace it with `else_value`.

The following example replaces all "race" IPUMS codes with 0 (white) or 1 (non-white). An IPUMS code of 100 is the "white" race category.

Maps T → U.

```
column_name = "race"
transforms = [
    {type = "when_value", value = 100, if_value = 0, else_value = 1}
]
```


### get_floor

Round down to the nearest whole number.

This example produces the general version of the IPUMS "relate" variable. The variable
is coded such that detailed categories are between the hundreds (300 is child of household
head, 301 is simply 'child', 302 is adopted child, 303 is step-child for instance).
The general categories are usually all that's needed (1 == household head, 2 == spouse,
3 == child, 4 == child-in-law, 5 == parent, 6 == parent-in-law, 7== sibling, 12 == not related to head).

Maps numerical → numerical.

```
[[column_mappings]]
alias = "relate_div_100"
column_name = "relate"
transforms = [
    {type = "divide_by_int", value = 100},
    {type = "get_floor"}
]
```
