"""
Módulo de Processamento

Este pacote expõe as funções de alto nível para as etapas de profiling e
limpeza de dados, criando uma API unificada para o pipeline de pré-processamento.
"""

from .profiling import (
    create_global_report,
    create_column_report,
    create_unique_values_report,
    create_numerical_report
)

from .cleaning import (
    apply_rename,
    apply_standardize_nulls,
    apply_data_types,
    apply_drop_duplicates,
    apply_strip_strings,
    apply_select_columns,
    apply_value_mapping
)