import functools
import pandas as pd

def validate_dataframe(func):
    """Decorator que valida se o primeiro argumento da função é um pandas DataFrame."""
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")
        return func(df, *args, **kwargs)
    return wrapper


def validate_column_type(allowed_type: str):
    """
    Decorator que valida se o 'column_name' de um método de classe de análise
    pertence ao tipo de coluna permitido ('numeric' ou 'categorical').
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, column_name, *args, **kwargs):
            if allowed_type == 'numeric' and column_name not in self.numeric_cols:
                return None
            if allowed_type == 'categorical' and column_name not in self.categorical_cols:
                return None
            return func(self, column_name, *args, **kwargs)
        return wrapper
    return decorator