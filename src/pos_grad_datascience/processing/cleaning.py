# C:\dev\nome-do-projeto\src\eda\data_cleaner.py

"""
Módulo de funções para limpeza e transformação estrutural de DataFrames.

Fornece um conjunto de operações de limpeza comuns, cada uma retornando
o DataFrame transformado junto com um relatório detalhado da operação.
"""

import logging
from typing import Dict, List, Set, Tuple
import re
import numpy as np
import pandas as pd

DEFAULT_NULL_PLACEHOLDERS: Set[str] = {
    '?', ' ?', 'NULL', '', ' ', 'NA', 'N/A', 'NaN', 'nan', 'None'
}





def apply_rename(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Padroniza nomes de colunas para o formato UPPER_SNAKE_CASE,
    convertendo CamelCase e substituindo caracteres especiais.

    Args:
        df (pd.DataFrame): O DataFrame original.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame renomeado e relatório.
    
    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    df_renamed = df.copy()
    original_names = df.columns.tolist()
    
    new_names = []
    for name in original_names:
        # 1. Substitui hífens ou pontos por underscore
        temp_name = re.sub(r'[.-]', '_', name)
        
        # 2. Insere um underscore antes de uma letra maiúscula (exceto no início)
        #    Isso converte 'BusinessTravel' para 'Business_Travel'
        temp_name = re.sub(r'(?<!^)(?=[A-Z])', '_', temp_name)
        
        # 3. Converte toda a string para maiúsculas e remove espaços
        new_names.append(temp_name.upper().strip())
    
    df_renamed.columns = new_names

    mapping_df = pd.DataFrame({
        'NOME_ORIGINAL': original_names, 
        'NOME_NOVO': new_names
    })
    
    report_df = mapping_df[
        mapping_df['NOME_ORIGINAL'] != mapping_df['NOME_NOVO']
    ].reset_index(drop=True)
    
    logging.info(f"{len(report_df)} colunas foram renomeadas para o padrão UPPER_SNAKE_CASE.")
    return df_renamed, report_df


def apply_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica tipos de dados especificados a colunas-alvo e gera um relatório completo
    para todas as colunas do DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame original.
        type_mapping (Dict[str, str]): Dicionário de {coluna: tipo_desejado}.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame com tipos alterados e relatório completo da operação.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    df_typed = df.copy()
    conversion_results = []
    
    logging.info(f"Iniciando tentativa de conversão de tipos para {len(type_mapping)} colunas.")

    # 1. Processa de forma eficiente apenas as colunas mapeadas
    for column, new_type in type_mapping.items():
        result_entry = {"NOME_COLUNA": column, "TIPO_DESEJADO": new_type}

        if column not in df_typed.columns:
            result_entry.update({"STATUS": "Falha", "MENSAGEM_ERRO": "Coluna não encontrada"})
            logging.warning(f"Coluna '{column}' do mapeamento não encontrada no DataFrame.")
            conversion_results.append(result_entry)
            continue

        original_type = str(df_typed[column].dtype)
        if original_type == new_type:
            result_entry.update({"STATUS": "Não Alterado (Tipo já era o correto)", "MENSAGEM_ERRO": ""})
        else:
            try:
                df_typed[column] = df_typed[column].astype(new_type)
                result_entry.update({"STATUS": "Sucesso", "MENSAGEM_ERRO": ""})
            except Exception as e:
                result_entry.update({"STATUS": "Falha na Conversão", "MENSAGEM_ERRO": str(e)})
                logging.warning(f"Falha ao converter '{column}' para '{new_type}'. Erro: {e}")
        
        conversion_results.append(result_entry)

    # 2. Cria o relatório completo
    report_df = pd.DataFrame({
        "NOME_COLUNA": df.columns,
        "TIPO_ORIGINAL": df.dtypes.astype(str)
    })
    
    # Adiciona os tipos finais após todas as conversões
    report_df['TIPO_FINAL'] = df_typed.dtypes.astype(str)

    # Se houver resultados de conversão, faz o merge
    if conversion_results:
        conversion_df = pd.DataFrame(conversion_results)
        report_df = pd.merge(report_df, conversion_df, on="NOME_COLUNA", how="left")

    # 3. Preenche o status das colunas não mapeadas e finaliza o relatório
    report_df['STATUS'] = report_df['STATUS'].fillna("Não Mapeado")
    report_df['MENSAGEM_ERRO'] = report_df['MENSAGEM_ERRO'].fillna("")
    report_df['TIPO_DESEJADO'] = report_df['TIPO_DESEJADO'].fillna("")
    
    logging.info("Conversão de tipos concluída.")
    return df_typed, report_df


def apply_select_columns(df: pd.DataFrame, columns_to_keep: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seleciona um subconjunto de colunas e gera um relatório."""
    df_selected = df[columns_to_keep]
    report_df = pd.DataFrame({"COLUNAS_MANTIDAS": columns_to_keep})
    logging.info(f"Seleção de colunas concluída. {len(columns_to_keep)} colunas mantidas.")
    return df_selected, report_df


def apply_drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove linhas duplicadas e gera um relatório da operação.

    Args:
        df (pd.DataFrame): DataFrame que pode conter duplicatas.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame sem duplicatas e relatório.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    lines_before = len(df)
    duplicate_count = df.duplicated().sum()
    
    df_deduplicated = df.drop_duplicates()
    lines_after = len(df_deduplicated)

    report_df = pd.DataFrame({
        "MÉTRICA": ["Linhas duplicatas", "Linhas antes", "Linhas após"],
        "VALOR": [duplicate_count, lines_before, lines_after]
    })
    
    logging.info(f"{duplicate_count} linhas duplicadas foram removidas.")
    return df_deduplicated, report_df


def apply_null_substitution(df: pd.DataFrame, null_placeholders: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Substitui placeholders por np.nan e gera um relatório da operação.

    Args:
        df (pd.DataFrame): DataFrame a ser limpo.
        null_placeholders (List[str], optional): Placeholders adicionais.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame com nulos padronizados e relatório.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    final_placeholders = DEFAULT_NULL_PLACEHOLDERS.copy()
    if null_placeholders:
        final_placeholders.update(null_placeholders)
    
    placeholders_list = list(final_placeholders)
    logging.info("Iniciando substituição para: %s", placeholders_list)

    nulls_before = df.isnull().sum()
    df_substituted = df.replace(placeholders_list, np.nan)
    nulls_after = df_substituted.isnull().sum()

    report_df = pd.DataFrame({
        'QT_NULOS_ANTES': nulls_before,
        'QT_NULOS_DEPOIS': nulls_after
    })
    report_df['%_NULOS_DEPOIS'] = (report_df['QT_NULOS_DEPOIS'] / len(df)) * 100 if len(df) > 0 else 0
    report_df['COLUNA_AFETADA'] = report_df['QT_NULOS_DEPOIS'] > report_df['QT_NULOS_ANTES']
    
    report_df = report_df.reset_index().rename(columns={'index': 'COLUNA'}).round(4)
    logging.info("Substituição de nulos concluída.")
    return df_substituted, report_df


def apply_strip_strings(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove espaços em branco de colunas de texto e gera um relatório.

    Args:
        df (pd.DataFrame): O DataFrame a ser limpo.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame limpo e relatório da operação.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    df_stripped = df.copy()
    report_data = []
    string_cols = df_stripped.select_dtypes(include=['object', 'category']).columns

    logging.info("Verificando %d colunas de texto para remoção de espaços.", len(string_cols))

    for col in df_stripped.columns:
        cells_changed = 0
        was_affected = False
        
        if col in string_cols:
            original_series = df_stripped[col]
            stripped_series = original_series.str.strip()
            
            # **LÓGICA CORRIGIDA PARA O BUG DE CONTAGEM**
            # Compara a série original com a nova, ignorando NaNs na contagem.
            cells_changed = (original_series.ne(stripped_series) & original_series.notna()).sum()

            if cells_changed > 0:
                df_stripped[col] = stripped_series
                was_affected = True
        
        report_data.append({
            "COLUNA": col,
            "TIPO_DADO": str(df_stripped[col].dtype),
            "CELULAS_ALTERADAS": cells_changed,
            "FOI_AFETADA": was_affected
        })

    report_df = pd.DataFrame(report_data)
    total_changes = report_df['CELULAS_ALTERADAS'].sum()
    logging.info("Limpeza de strings concluída. Total de %d células alteradas.", total_changes)
    
    return df_stripped, report_df


def apply_value_mapping(df: pd.DataFrame, mapping_dict: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica um mapeamento de valores a colunas específicas e gera um relatório.

    Args:
        df (pd.DataFrame): O DataFrame a ser processado.
        mapping_dict (Dict[str, Dict]): Dicionário {coluna: {de: para}}.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame mapeado e relatório.
        
    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    df_mapped = df.copy()
    report_data = []
    logging.info("Iniciando mapeamento de valores para %d colunas.", len(mapping_dict))

    for col_name, value_map in mapping_dict.items():
        if col_name in df_mapped.columns:
            # Gera relatório antes de modificar
            for from_val, to_val in value_map.items():
                affected_cells = df_mapped[col_name].eq(from_val).sum()
                if affected_cells > 0:
                    report_data.append({
                        "COLUNA_AFETADA": col_name,
                        "VALOR_ORIGINAL": from_val,
                        "VALOR_NOVO": to_val,
                        "QUANTIDADE_ALTERACOES": affected_cells
                    })
            # Aplica a substituição
            df_mapped[col_name] = df_mapped[col_name].replace(value_map)
        else:
            logging.warning("Coluna '%s' do mapeamento não encontrada no DataFrame.", col_name)

    report_df = pd.DataFrame(report_data)
    logging.info("Mapeamento de valores concluído.")
    return df_mapped, report_df