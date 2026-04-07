import pandas as pd



# Drop unwanted columns
def drop_columns(data: pd.DataFrame, column_names: list) -> pd.DataFrame:

    """Picks only specified columns from DataFrame."""

    return data[column_names]



# Drop null values
def drop_nulls(data: pd.DataFrame) -> pd.DataFrame:

    """Drops rows with null values."""

    return data.dropna()



# Summary
def check_quality(data: pd.DataFrame):

    """Basic Quality Check. Returns DataFrame with columns, types 
    and unique, missing and duplicate elements plus their ratios.
    Complement/extension of .info() function."""

    n = len(data)

    summary = []
    for column in data:
        col_type = type(data[column].iloc[0])
        elements = data[column].count()
        unique = data[column].nunique(dropna = True)
        rate_unique = round(unique / n * 100, 2)
        missing = data[column].isna().sum() 
        rate_missing = round(missing / n * 100, 2)
        duplicated = data[column].duplicated().sum()
        
        summary.append((column, col_type, elements, unique, rate_unique, missing, rate_missing, duplicated))
    
    return pd.DataFrame(summary, columns = ['column', 'type', 'elements', 'unique', 'rate_unique', 'missing', 'rate_missing', 'duplicated'])
