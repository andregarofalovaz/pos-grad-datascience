# src

"""Módulo de profiling para diagnóstico inicial e análise de qualidade de DataFrames.

As ferramentas aqui contidas geram relatórios sistemáticos e detalhados sobre 
a estrutura, qualidade, distribuição e características físicas dos dados, 
servindo como a fundação para as etapas subsequentes de limpeza, EDA e modelagem.

As funções são projetadas para serem modulares e retornam DataFrames 'tidy',
facilitando a análise programática e a integração com pipelines automatizados.

Principais Funções:
-------------------
create_global_report:
    Gera uma visão macroscópica do DataFrame, com métricas de alto nível
    como dimensões, duplicatas, nulos e uso de memória.

create_column_report:
    Produz um relatório detalhado para cada coluna, focando em diagnóstico de
    qualidade de dados (nulos), estrutura (tipo de dado) e cardinalidade.

create_unique_values_report:
    Realiza uma análise aprofundada da distribuição de colunas categóricas,
    identificando nulos (padrão e ocultos) e as categorias mais frequentes.

create_numerical_report:
    Gera um diagnóstico técnico para colunas numéricas, focando em otimização
    de memória e validação de integridade (presença de negativos, zeros, etc.).

"""

import pandas as pd
import numpy as np
import functools
from typing import List, Set, Optional

from pos_grad_datascience.core.decorators import validate_dataframe


DEFAULT_NULL_PLACEHOLDERS: Set[str] = {
    '?', 'NULL', '', ' ', 'NA', 'N/A', 'NaN', 'nan', 'None'
}

@validate_dataframe_input
def create_global_report(df: pd.DataFrame) -> pd.DataFrame:
    """Gera um relatório global com métricas sobre um DataFrame.

    Esta função calcula um conjunto de métricas estatísticas e estruturais
    essenciais para uma visão geral rápida do DataFrame. A função é orientada
    por uma configuração interna, o que a torna facilmente extensível para
    novas métricas no futuro.

    Args:
        df (pd.DataFrame): O DataFrame de entrada a ser analisado.

    Returns:
        pd.DataFrame: Um DataFrame de relatório contendo duas colunas:
            - MÉTRICA: O nome da métrica calculada (e.g., 'Total de Linhas');
            - VALOR: O resultado da métrica, formatado para melhor legibilidade.
            
            As métricas padrão incluem totais de linhas, colunas, duplicatas,
            células nulas, uso de memória e a contagem de colunas por tipo de dado.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """

    # Define as métricas a serem calculadas, suas funções e formatos de saída.
    METRICS_CONFIG = [
        {"name": "Total de Linhas", "func": lambda d: d.shape[0], "format": "{:,.0f}"},
        {"name": "Total de Colunas", "func": lambda d: d.shape[1], "format": "{:,.0f}"},
        {"name": "Linhas Duplicadas", "func": lambda d: d.duplicated().sum(), "format": "{:,.0f}"},
        {"name": "Células Nulas (Total)", "func": lambda d: d.isnull().sum().sum(), "format": "{:,.0f}"},
        {"name": "% de Células Nulas", "func": lambda d: (d.isnull().sum().sum() / d.size) * 100 if d.size > 0 else 0, "format": "{:.1f}%"},
        {"name": "Uso de Memória (KB)", "func": lambda d: d.memory_usage(deep=True).sum() / 1024, "format": "{:,.1f} KB"}
    ]

    # Adiciona à configuração as métricas de contagem para cada dtype presente no df.
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        METRICS_CONFIG.append(
            {"name": f"Colunas do tipo '{str(dtype)}'", "func": lambda d, c=count: c, "format": "{:,.0f}"}
        )

    # Itera sobre a configuração, executa cada função e formata o resultado.
    report_data = []
    for metric in METRICS_CONFIG:
        value = metric["func"](df)
        report_data.append({
            "MÉTRICA": metric["name"],
            "VALOR_BRUTO": value,
            "VALOR": metric["format"].format(value)
        })
    
    final_df = pd.DataFrame(report_data)
    
    # Retorna o DataFrame final contendo apenas as colunas de exibição
    return final_df[['MÉTRICA', 'VALOR']]

@validate_dataframe_input
def create_column_report(df: pd.DataFrame) -> pd.DataFrame:
    """Gera um relatório estatístico detalhado para cada coluna de um DataFrame.

    Esta função analisa um DataFrame e produz um relatório de profiling, onde
    cada linha corresponde a uma coluna do DataFrame original. O relatório inclui
    informações sobre tipo de dado, contagem de nulos, cardinalidade e uso de
    memória, com formatação otimizada para leitura.

    Args:
        df (pd.DataFrame): O DataFrame de entrada a ser analisado.

    Returns:
        pd.DataFrame: Um DataFrame de relatório contendo as seguintes colunas:
            - COLUNA: O nome da coluna original;
            - TIPO: O tipo de dado (dtype) da coluna;
            - QT_NULOS: A quantidade absoluta de valores nulos;
            - %_NULOS: O percentual de valores nulos;
            - QTD_VALORES_UNICOS: A quantidade de valores únicos (cardinalidade);
            - %_VALORES_UNICOS: O percentual de valores únicos;
            - USO_MEMORIA_KB: O uso de memória da coluna em KB.

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """

    # Calcula o uso de memória por coluna (em KB)
    memory_usage_kb = df.memory_usage(deep=True, index=True) / 1024
    memory_usage_kb = memory_usage_kb.drop('Index', errors='ignore')

    # Cria o relatório a partir de um dicionário de Series calculadas
    report = pd.DataFrame({
        'TIPO': df.dtypes.astype(str),
        'QT_NULOS': df.isnull().sum(),
        '%_NULOS': df.isnull().mean() * 100,
        'QTD_VALORES_UNICOS': df.nunique(),
        '%_VALORES_UNICOS': (df.nunique() / len(df)) * 100 if len(df) > 0 else 0,
        'USO_MEMORIA_KB': memory_usage_kb
    })
    
    # Formata as colunas para apresentação final
    report['%_NULOS'] = report['%_NULOS'].round(1)
    report['%_VALORES_UNICOS'] = report['%_VALORES_UNICOS'].round(1)
    report['USO_MEMORIA_KB'] = report['USO_MEMORIA_KB'].round(1)

    # Finaliza e retorna o relatório
    return report.reset_index().rename(columns={'index': 'COLUNA'})

@validate_dataframe_input
def create_unique_values_report(
    df: pd.DataFrame,
    column_types: List[str],
    top_n: Optional[int] = 15,
    null_placeholders: Optional[List[str]] = None
) -> pd.DataFrame:
    """Gera um relatório avançado de distribuição de valores únicos.

    Esta função analisa colunas categóricas e retorna um relatório tidy com as 'n' 
    categorias mais frequentes, priorizando a exibição de valores nulos e
    agrupando as demais categorias em uma linha de resumo.

    Args:
        df (pd.DataFrame): O DataFrame a ser analisado.
        column_types (List[str]): Tipos de dados a serem perfilados (e.g., ['object']).
        top_n (Optional[int]): Número de categorias mais frequentes a serem exibidas.
                               Se None, exibe todas. Default é 15.
        null_placeholders (Optional[List[str]]): Placeholders adicionais de nulos.

    Returns:
        pd.DataFrame: DataFrame 'tidy' com as colunas na ordem:
                      ['COLUNA', 'VALOR_UNICO', 'CONTAGEM', 
                       'PERCENTUAL', 'POSSIVEL_NULO'].
    """

    final_placeholders = DEFAULT_NULL_PLACEHOLDERS.copy()
    if null_placeholders:
        final_placeholders.update(null_placeholders)

    target_cols = df.select_dtypes(include=column_types).columns
    report_data = []

    for col in target_cols:
        counts = df[col].value_counts(dropna=False)
        total_count = len(df[col])
        percentages = (counts / total_count) * 100 if total_count > 0 else counts * 0
        
        null_values_data = []
        valid_counts = counts.copy()

        for value, count in counts.items():
            is_placeholder = str(value) in final_placeholders
            is_nan = pd.isnull(value)

            if is_placeholder or is_nan:
                null_values_data.append({
                    'COLUNA': col,
                    'POSSIVEL_NULO': True,
                    'VALOR_UNICO': value,
                    'CONTAGEM': count,
                    'PERCENTUAL': percentages[value]
                })
                valid_counts.drop(value, inplace=True)
        
        report_data.extend(null_values_data)

        if top_n is not None and len(valid_counts) > top_n:
            top_n_counts = valid_counts.head(top_n)
            others_count = valid_counts.iloc[top_n:].sum()
            others_perc = percentages.loc[valid_counts.iloc[top_n:].index].sum()
        else:
            top_n_counts = valid_counts
            others_count = 0
        
        for value, count in top_n_counts.items():
            report_data.append({
                'COLUNA': col, 'POSSIVEL_NULO': False,
                'VALOR_UNICO': value, 'CONTAGEM': count, 'PERCENTUAL': percentages[value]
            })

        if others_count > 0:
            report_data.append({
                'COLUNA': col, 'POSSIVEL_NULO': False,
                'VALOR_UNICO': '[Valores Fora do Top N]',
                'CONTAGEM': others_count, 'PERCENTUAL': others_perc
            })

    # Cria o DataFrame com os dados coletados
    final_report = pd.DataFrame(report_data)

    # Formata e reordena o DataFrame final
    if not final_report.empty:
        final_report['PERCENTUAL'] = final_report['PERCENTUAL'].round(1)
        
        # 2. Reordena as colunas para a sequência desejada
        final_report = final_report[[
            'COLUNA', 'VALOR_UNICO', 'CONTAGEM', 'PERCENTUAL', 'POSSIVEL_NULO'
        ]]
        
    return final_report

def _get_optimal_numeric_type(col: pd.Series) -> str:
    """Função auxiliar para encontrar o subtipo numérico mais eficiente."""
    col_min, col_max = col.min(), col.max()
    
    if pd.api.types.is_integer_dtype(col):
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            return 'int8'
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            return 'int16'
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            return 'int32'
        return 'int64'
    
    elif pd.api.types.is_float_dtype(col):
        # A conversão para float16 pode levar a perda de precisão, float32 é a otimização mais segura.
        try:
            if np.all(np.finfo(np.float32).min < col.dropna() < np.finfo(np.float32).max):
                return 'float32'
        except: # Lida com casos onde a coluna está vazia após o dropna
            return 'float32'
        return 'float64'
    
    return str(col.dtype)

@validate_dataframe_input
def create_numerical_report(df: pd.DataFrame) -> pd.DataFrame:
    """Gera um relatório de diagnóstico físico e de qualidade para colunas numéricas.

    Esta função analisa as colunas numéricas de um DataFrame e produz um relatório
    com métricas focadas em limpeza de dados e otimização de memória. Cada linha
    do relatório corresponde a uma coluna numérica do DataFrame original.

    Args:
        df (pd.DataFrame): O DataFrame a ser analisado.

    Returns:
        pd.DataFrame: Um DataFrame de relatório contendo as seguintes colunas:
            - COLUNA: O nome da coluna numérica.
            - TIPO_ATUAL: O tipo de dado (dtype) atual da coluna.
            - VALOR_MINIMO: O valor mínimo presente na coluna.
            - VALOR_MAXIMO: O valor máximo presente na coluna.
            - TIPO_OTIMIZADO: A sugestão do subtipo numérico mais eficiente em memória.
            - TEM_NEGATIVOS: Flag booleana (True) se existirem valores negativos.
            - %_ZEROS: O percentual de valores que são exatamente zero.
            - PODE_SER_INTEIRO: Flag booleana (True) se uma coluna float contém
                                apenas valores inteiros (e.g., 1.0, 2.0).

    Raises:
        TypeError: Se o objeto de entrada não for um pandas DataFrame.
    """

    # Seleciona apenas as colunas numéricas para a análise
    numerical_cols = df.select_dtypes(include=np.number).columns
    report_data = []

    # Itera sobre cada coluna numérica para gerar suas métricas
    for col in numerical_cols:
        
        can_be_int = False
        # Verifica se a coluna float pode ser convertida para inteiro
        if pd.api.types.is_float_dtype(df[col]):
            # Compara todos os valores não nulos com suas versões arredondadas
            can_be_int = (df[col].dropna() == df[col].dropna().round()).all()

        # Coleta todas as métricas para a coluna atual
        report_data.append({
            'COLUNA': col,
            'TIPO_ATUAL': str(df[col].dtype),
            'VALOR_MINIMO': df[col].min(),
            'VALOR_MAXIMO': df[col].max(),
            'TIPO_OTIMIZADO': _get_optimal_numeric_type(df[col]),
            'TEM_NEGATIVOS': (df[col] < 0).any(),
            '%_ZEROS': (df[col] == 0).mean() * 100,
            'PODE_SER_INTEIRO': can_be_int
        })

    # Cria e formata o DataFrame final
    final_report = pd.DataFrame(report_data)
    if not final_report.empty:
        final_report['%_ZEROS'] = final_report['%_ZEROS'].round(1)
    
    return final_report