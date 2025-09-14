# src/pos_grad_datascience/core/utils.py

"""Módulo de funções utilitárias para o projeto.

Este módulo contém funções auxiliares que fornecem suporte a várias
etapas do pipeline, como configuração de ambiente, logging e validações.
"""

import sys
from pathlib import Path

def display_library_versions(requirements_path: str) -> str:
    """Lê um arquivo requirements.txt e gera uma tabela formatada com as versões.

    Esta função lê um arquivo de dependências, extrai os nomes e versões
    das bibliotecas e os apresenta em uma tabela de texto formatada,
    semelhante a uma tabela de banco de dados, incluindo a versão do Python
    em execução.

    Args:
        requirements_path (str): O caminho para o arquivo requirements.txt.

    Returns:
        str: Uma string multi-linha contendo a tabela formatada, pronta
             para ser impressa ou registrada em um log. Retorna uma mensagem
             de erro se o arquivo não for encontrado.
    """
    req_file = Path(requirements_path)
    if not req_file.is_file():
        return f"ERRO: Arquivo requirements.txt não encontrado em '{requirements_path}'."

    libraries = []
    # Lê o arquivo e extrai nome e versão de cada biblioteca
    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Usa .partition() para dividir de forma segura na primeira ocorrência de '=='
                name, separator, version = line.partition('==')
                if not separator: # Lida com linhas sem versão explícita
                    name = line
                    version = 'N/A'
                libraries.append((name, version))

    # Constrói a string da tabela, linha por linha
    bibliotecas_str = ""
    for name, version in sorted(libraries):
        bibliotecas_str += f"║ {name:<23}║ {version:>12} ║\n"

    # Pega a versão do Python do ambiente atual
    py_version_str = ".".join(map(str, sys.version_info[:3]))
    versao_python = f"Versão do Python: {py_version_str}"
    largura_total = 35

    # Monta a estrutura final da tabela
    mensagem = (
        "\n"
        "╔════════════════════════╦══════════════╗\n"
        "║       Biblioteca       ║    Versão    ║\n"
        "╠════════════════════════╬══════════════╣\n"
      f"{bibliotecas_str}"
        "╠════════════════════════╩══════════════╣\n"
        f"║  {versao_python:^{largura_total}}  ║\n"
        "╚═══════════════════════════════════════╝"
        "\n"
    )

    return mensagem