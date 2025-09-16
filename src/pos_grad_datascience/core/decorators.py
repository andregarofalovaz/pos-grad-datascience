import functools
import pandas as pd

def validate_dataframe_input(func):
    """Decorator que valida se o primeiro argumento da função é um pandas DataFrame."""
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")
        return func(df, *args, **kwargs)
    return wrapper