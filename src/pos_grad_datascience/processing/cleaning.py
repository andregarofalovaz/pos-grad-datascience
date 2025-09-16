"""Módulo de funções para limpeza e transformação estrutural de DataFrames.

Este módulo contém a coleção de funções para a Etapa 2 do pipeline de análise
de dados: Limpeza dos Dados. Cada função realiza uma operação de transformação
específica no DataFrame (e.g., renomear colunas, padronizar nulos, remover
duplicatas) e, crucialmente, retorna tanto o DataFrame modificado quanto um
relatório de auditoria detalhado da operação realizada.

As funções são projetadas para serem parametrizáveis e integradas a um
pipeline orquestrado, garantindo um processo de limpeza de dados
reprodutível, auditável e robusto.

Principais Funções:
-------------------
apply_rename:
    Padroniza os nomes das colunas para o formato UPPER_SNAKE_CASE.

apply_standardize_nulls:
    Converte múltiplos formatos de nulos ocultos (placeholders) para o
    formato padrão do pandas (np.nan).

apply_data_types:
    Aplica tipos de dados específicos a um conjunto de colunas, com base
    em um dicionário de mapeamento.

apply_drop_duplicates:
    Remove linhas duplicadas, com opções para definir o escopo da verificação
    (subset) e a estratégia de manutenção (keep).

apply_strip_strings:
    Remove espaços em branco do início e do fim de todas as colunas de texto.

apply_select_columns:
    Seleciona ou remove colunas do DataFrame.

apply_value_mapping:
    Substitui valores específicos em colunas categóricas para padronização.

"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import re
import numpy as np
import pandas as pd

from pos_grad_datascience.core.decorators import validate_dataframe

DEFAULT_NULL_PLACEHOLDERS: Set[str] = {
    '?', ' ?', 'NULL', '', ' ', 'NA', 'N/A', 'NaN', 'nan', 'None'
}


@validate_dataframe_input
def apply_rename(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Padroniza os nomes das colunas de um DataFrame para o formato UPPER_SNAKE_CASE.

    Esta função aplica uma série de transformações para converter os nomes das
    colunas em um padrão consistente. A lógica lida com a remoção de espaços,
    a substituição de caracteres especiais (e.g., '.', '-') por underscores, e a
    conversão inteligente de 'CamelCase' para 'snake_case' antes de passar o
    resultado final para maiúsculas.

    Args:
        df (pd.DataFrame): O DataFrame original cujas colunas serão renomeadas.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Uma tupla contendo:
            - df_renamed: O DataFrame com os nomes das colunas padronizados.
            - report_df: Um relatório em DataFrame que lista apenas as colunas
                         cujos nomes foram efetivamente alterados, mostrando
                         o nome original e o novo nome.

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
        
        # 2. Insere um underscore antes de uma letra maiúscula APENAS se precedida por uma minúscula
        temp_name = re.sub(r'(?<=[a-z])([A-Z])', r'_\1', temp_name)
        
        # 3. Converte toda a string para maiúsculas e remove espaços
        new_names.append(temp_name.upper().strip())
    
    df_renamed.columns = new_names

    report_data = [
        {"NOME_ORIGINAL": orig, "NOME_NOVO": new}
        for orig, new in zip(original_names, new_names)
        if orig != new
    ]
    report_df = pd.DataFrame(report_data)
    
    logging.info(f"{len(report_df)} colunas foram renomeadas para o padrão UPPER_SNAKE_CASE.")
    return df_renamed, report_df

@validate_dataframe_input
def apply_standardize_nulls(df: pd.DataFrame, null_placeholders: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Padroniza valores nulos ocultos (placeholders) para o formato np.nan.

    Esta função varre o DataFrame em busca de representações não padronizadas de
    valores ausentes e as converte para o formato padrão do pandas (np.nan).
    Ela utiliza uma lista de placeholders padrão definida na constante
    `DEFAULT_NULL_PLACEHOLDERS` deste módulo, que pode ser estendida através
    do argumento `null_placeholders`.

    Args:
        df (pd.DataFrame): DataFrame a ser padronizado.
        null_placeholders (List[str], optional): Placeholders adicionais para se 
                                                 juntar à lista padrão.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame com nulos padronizados e relatório.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    final_placeholders = DEFAULT_NULL_PLACEHOLDERS.copy()
    if null_placeholders:
        final_placeholders.update(null_placeholders)
    
    placeholders_list = list(final_placeholders)
    logging.info("Iniciando padronização para: %s", placeholders_list)

    nulls_before = df.isnull().sum()
    df_standardized = df.replace(placeholders_list, np.nan)
    nulls_after = df_standardized.isnull().sum()

    report_df = pd.DataFrame({
        'QT_NULOS_ANTES': nulls_before,
        'QT_NULOS_DEPOIS': nulls_after
    })
    report_df['%_NULOS_DEPOIS'] = (report_df['QT_NULOS_DEPOIS'] / len(df)) * 100 if len(df) > 0 else 0
    report_df['COLUNA_AFETADA'] = report_df['QT_NULOS_DEPOIS'] > report_df['QT_NULOS_ANTES']
    
    report_df = report_df.reset_index().rename(columns={'index': 'COLUNA'}).round(4)
    logging.info("Padronização de nulos concluída.")
    return df_standardized, report_df

@validate_dataframe_input
def apply_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica tipos de dados especificados a colunas e gera um relatório completo.

    Esta função itera sobre as colunas de um DataFrame e tenta converter aquelas
    especificadas em um dicionário de mapeamento para o tipo de dado desejado.
    Ela é robusta a erros de conversão e a colunas inexistentes no mapeamento.

    Args:
        df (pd.DataFrame): O DataFrame original.
        type_mapping (Dict[str, str]): Dicionário no formato {nome_da_coluna: tipo_desejado}.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_typed: O DataFrame com os tipos de dados alterados.
            - report_df: Um relatório completo da operação, com o status para cada
                         coluna do DataFrame original.
    
    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    df_typed = df.copy()
    report_data = []
    
    logging.info(f"Iniciando verificação e conversão de tipos para {len(df.columns)} colunas.")

    # Itera sobre todas as colunas para construir um relatório completo
    for column in df.columns:
        original_type = str(df[column].dtype)
        
        # Dicionário para armazenar o resultado desta coluna
        result_entry = {
            "NOME_COLUNA": column,
            "TIPO_ORIGINAL": original_type,
            "TIPO_DESEJADO": "",
            "STATUS": "Não Mapeado",
            "MENSAGEM_ERRO": ""
        }

        # Verifica se a coluna atual está no dicionário de mapeamento
        if column in type_mapping:
            new_type = type_mapping[column]
            result_entry["TIPO_DESEJADO"] = new_type

            if original_type == new_type:
                result_entry["STATUS"] = "Não Alterado (Tipo já era o correto)"
            else:
                try:
                    df_typed[column] = df_typed[column].astype(new_type)
                    result_entry["STATUS"] = "Sucesso"
                except Exception as e:
                    result_entry["STATUS"] = "Falha na Conversão"
                    result_entry["MENSAGEM_ERRO"] = str(e)
                    logging.warning(f"Falha ao converter '{column}' para '{new_type}'. Erro: {e}")
        
        # Adiciona o tipo final ao relatório (pode ter mudado ou não)
        result_entry["TIPO_FINAL"] = str(df_typed[column].dtype)
        report_data.append(result_entry)

    report_df = pd.DataFrame(report_data)
    logging.info("Conversão de tipos concluída.")
    
    return df_typed, report_df

@validate_dataframe_input
def apply_drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove linhas duplicadas de um DataFrame e gera um relatório da operação.

    Esta função é um wrapper para a função `pandas.DataFrame.drop_duplicates`,
    permitindo a especificação de um subconjunto de colunas para a identificação
    de duplicatas e a estratégia de qual registro manter.

    Args:
        df (pd.DataFrame): O DataFrame de entrada que pode conter duplicatas.
        subset (Optional[List[str]], optional): Lista de nomes de colunas a serem
            consideradas para identificar duplicatas. Se None, todas as colunas
            são utilizadas. O padrão é None.
        keep (str, optional): O método para tratar as duplicatas:
            - 'first': Mantém a primeira ocorrência (padrão).
            - 'last': Mantém a última ocorrência.
            - False: Remove todas as ocorrências de duplicatas.
            O padrão é 'first'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - df_deduplicated: O DataFrame sem as linhas duplicadas.
            - report_df: Um relatório em DataFrame que sumariza a operação.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
        ValueError: Se o valor de 'keep' for inválido.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")
    if keep not in ['first', 'last', False]:
        raise ValueError("O parâmetro 'keep' deve ser 'first', 'last' ou False.")

    lines_before = len(df)
    # Calcula a contagem de duplicatas com base nos mesmos parâmetros
    duplicate_count = df.duplicated(subset=subset, keep=keep).sum()
    
    # Remove as duplicatas
    df_deduplicated = df.drop_duplicates(subset=subset, keep=keep)
    lines_after = len(df_deduplicated)

    # Cria o relatório da operação
    report_df = pd.DataFrame({
        "MÉTRICA": [
            "Critério de Colunas", 
            "Estratégia (Keep)",
            "Linhas Duplicadas Removidas", 
            "Linhas Antes da Remoção", 
            "Linhas Após a Remoção"
        ],
        "VALOR": [
            'Todas as Colunas' if subset is None else str(subset),
            str(keep),
            duplicate_count, 
            lines_before, 
            lines_after
        ]
    })
    
    logging.info(f"{duplicate_count} linhas duplicadas foram removidas.")
    return df_deduplicated, report_df

@validate_dataframe_input
def apply_strip_strings(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove espaços em branco do início e fim de colunas de texto.

    Esta função itera sobre todas as colunas do tipo 'object' ou 'category',
    aplica a remoção de espaços em branco (strip) e gera um relatório
    listando apenas as colunas que foram efetivamente modificadas.

    Args:
        df (pd.DataFrame): O DataFrame a ser limpo.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - df_stripped: O DataFrame com as strings limpas.
            - report_df: Um relatório das colunas que sofreram alterações.
    
    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")

    df_stripped = df.copy()
    report_data = []
    
    # 1. Itera de forma eficiente apenas sobre as colunas de interesse
    string_cols = df_stripped.select_dtypes(include=['object', 'category']).columns
    logging.info(f"Verificando {len(string_cols)} colunas de texto para remoção de espaços.")

    for col in string_cols:
        # Pula colunas que não contêm dados do tipo string (caso raro, mas seguro)
        if not hasattr(df_stripped[col].str, 'strip'):
            continue
            
        original_series = df_stripped[col].copy()
        stripped_series = original_series.str.strip()
        
        # Lógica robusta para contar apenas as alterações reais, ignorando NaNs
        cells_changed = (original_series.ne(stripped_series) & original_series.notna()).sum()

        # 2. Adiciona ao relatório APENAS se houveram alterações
        if cells_changed > 0:
            df_stripped[col] = stripped_series
            
            report_data.append({
                "COLUNA": col,
                "CELULAS_ALTERADAS": cells_changed,
            })

    report_df = pd.DataFrame(report_data)
    total_changes = report_df['CELULAS_ALTERADAS'].sum() if not report_df.empty else 0
    logging.info(f"Limpeza de strings concluída. Total de {total_changes} células alteradas.")
    
    return df_stripped, report_df

@validate_dataframe_input
def apply_select_columns(df: pd.DataFrame, columns_to_keep: Optional[List[str]] = None, columns_to_drop: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seleciona ou remove colunas de um DataFrame e gera um relatório.

    A função opera em um de dois modos:
    1. Se `columns_to_keep` for fornecido, apenas estas colunas permanecerão.
    2. Se `columns_to_drop` for fornecido, estas colunas serão removidas.

    Apenas um dos dois parâmetros pode ser utilizado por vez.

    Args:
        df (pd.DataFrame): O DataFrame original.
        columns_to_keep (Optional[List[str]], optional): Lista de colunas a serem mantidas.
        columns_to_drop (Optional[List[str]], optional): Lista de colunas a serem removidas.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_processed: O DataFrame com o subconjunto de colunas.
            - report_df: Um relatório de auditoria da operação.
    
    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
        ValueError: Se ambos ou nenhum dos parâmetros de seleção for fornecido.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")
    if (columns_to_keep is None) == (columns_to_drop is None):
        raise ValueError("Exatamente um dos parâmetros 'columns_to_keep' ou 'columns_to_drop' deve ser fornecido.")

    cols_before = df.columns.tolist()
    
    if columns_to_keep is not None:
        # Valida se todas as colunas a manter existem
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            logging.warning(f"As seguintes colunas não foram encontradas e serão ignoradas: {missing_cols}")
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        df_processed = df[columns_to_keep]
        operation_type = "Manter Colunas"
    else: # columns_to_drop is not None
        df_processed = df.drop(columns=columns_to_drop, errors='ignore')
        operation_type = "Remover Colunas"

    cols_after = df_processed.columns.tolist()
    cols_dropped = list(set(cols_before) - set(cols_after))

    report_df = pd.DataFrame({
        "MÉTRICA": ["Operação Realizada", "Colunas Antes", "Colunas Depois", "Colunas Removidas"],
        "VALOR": [operation_type, len(cols_before), len(cols_after), len(cols_dropped)],
        "DETALHE": ["-", ", ".join(cols_before), ", ".join(cols_after), ", ".join(cols_dropped)]
    })

    logging.info(f"{len(cols_dropped)} colunas foram removidas.")
    return df_processed, report_df

@validate_dataframe_input
def apply_value_mapping(df: pd.DataFrame, mapping_dict: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica um mapeamento de valores a colunas específicas e gera um relatório.

    Esta função substitui valores em colunas do DataFrame com base em um
    dicionário de mapeamento fornecido. É ideal para padronizar categorias
    ou corrigir valores específicos de forma controlada e auditável.

    Args:
        df (pd.DataFrame): O DataFrame a ser processado.
        mapping_dict (Dict[str, Dict]): Dicionário aninhado no formato
            {nome_da_coluna: {valor_original: valor_novo}}.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - df_mapped: O DataFrame com os valores mapeados.
            - report_df: Um relatório detalhado das alterações realizadas.
        
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
            
            original_counts = df_mapped[col_name].value_counts()
            
            for from_val, to_val in value_map.items():
                affected_cells = original_counts.get(from_val, 0)
                if affected_cells > 0:
                    report_data.append({
                        "COLUNA_AFETADA": col_name,
                        "VALOR_ORIGINAL": from_val,
                        "VALOR_NOVO": to_val,
                        "QUANTIDADE_ALTERACOES": affected_cells
                    })
            
            if pd.api.types.is_categorical_dtype(df_mapped[col_name]):
                logging.info(f"Coluna '{col_name}' é categórica. Usando .cat.rename_categories().")
                df_mapped[col_name] = df_mapped[col_name].cat.rename_categories(value_map)
            else:
                logging.info(f"Coluna '{col_name}' é do tipo object. Usando .replace().")
                df_mapped[col_name] = df_mapped[col_name].replace(value_map)
        else:
            logging.warning("Coluna '%s' do mapeamento não encontrada no DataFrame.", col_name)

    report_df = pd.DataFrame(report_data)
    logging.info("Mapeamento de valores concluído.")
    return df_mapped, report_df