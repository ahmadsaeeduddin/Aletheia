import pandas as pd
import pyreadstat

df, meta = pyreadstat.read_sav("bh.sav")

# You can now use pandas to analyze the data
print(df.head())
print(df.columns)
print(meta.number_columns)        # Total number of columns
print(meta.variable_value_labels) # Dictionary of value labels
print(meta.file_label)            # File-level label (if present)
