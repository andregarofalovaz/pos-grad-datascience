# log_configurator.py

"""
Módulo de configuração de logging para a aplicação.

Este módulo fornece:
1. Uma função 'setup_custom_logging' para configurar o logger raiz do Python
   com um formato colorido, padronizado e com nível de verbosidade controlável.
2. Uma classe 'IndentedLogger' que encapsula o logger padrão para adicionar
   funcionalidade de indentação hierárquica automática, melhorando a
   legibilidade de processos com etapas aninhadas.
"""

import logging
import colorlog
import time


# ==============================================================================
# SETUP GLOBAL DO LOGGER
# ==============================================================================
def setup_custom_logging(level: str = 'INFO'):
    """
    Configura o logger raiz para toda a aplicação Python.

    Esta função deve ser chamada apenas uma vez, no ponto de entrada da aplicação,
    para garantir um comportamento de logging consistente. Ela define o formato,
    as cores, o nível de verbosidade e silencia bibliotecas de terceiros.

    :param level: O nível de log desejado como string (ex: 'DEBUG', 'INFO').
                  O padrão é 'INFO' para uma saída mais limpa em produção.
    """
    # Mapeamento seguro de strings de nível para as constantes numéricas do módulo logging.
    # Isso permite que a configuração seja feita com strings amigáveis (ex: em um YAML).
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    # .get() com um valor padrão garante que, se um nível inválido for fornecido,
    # a aplicação não quebre e assuma um padrão seguro (INFO).
    numeric_level = log_level_map.get(level.upper(), logging.INFO)

    # Obter o logger raiz (sem nome) significa que esta configuração se aplicará
    # a todos os loggers da aplicação, a menos que eles tenham uma configuração específica.
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Linha CRÍTICA: Limpa quaisquer 'handlers' (destinos de log) pré-existentes.
    # Essencial para evitar mensagens duplicadas em ambientes como notebooks (Jupyter)
    # ou ao recarregar módulos interativamente.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Utiliza o StreamHandler do colorlog para direcionar a saída para o console (stderr).
    handler = colorlog.StreamHandler()

    # Define o formato e as cores para cada nível de log.
    formatter = colorlog.ColoredFormatter(
        # Formato da mensagem:
        # %(log_color)s     -> Aplica a cor definida para o nível do log.
        # %(asctime)s       -> Data e hora da mensagem.
        # %(levelname)-8s   -> Nível do log (INFO, DEBUG), com padding de 8 caracteres.
        # %(name)-16s       -> Nome do logger que emitiu a mensagem (ex: PipelineManager), com padding de 16 caracteres.
        # %(message)s       -> A mensagem de log principal.
        '%(log_color)s%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s',
        datefmt='%d/%m/%y | %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_yellow,bg_black',
        }
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silencia bibliotecas de terceiros que são excessivamente verbosas,
    # definindo seu nível de log para WARNING, sem afetar o nível da nossa aplicação.
    logging.getLogger('py4j').setLevel(logging.WARNING)
    logging.getLogger('pyspark').setLevel(logging.WARNING)


# ==============================================================================
# CLASSE DE LOGGER CUSTOMIZADO
# ==============================================================================
class IndentedLogger:
    """
    Um wrapper sobre o logger padrão do Python para fornecer indentação
    hierárquica automática, ideal para visualizar processos com sub-etapas.
    """

    def __init__(self, name: str, indent_char: str = "    "):
        """
        Inicializa o logger.

        :param name: O nome do logger, geralmente __name__ ou o nome da classe.
                     Isso permite o controle fino da saída por módulo/classe.
        :param indent_char: O caractere ou string a ser usado para cada nível de indentação.
        """
        self.logger = logging.getLogger(name)
        self.indent_level = 0
        self.indent_char = indent_char

    def _log(self, level, msg, *args, **kwargs):
        """
        Método privado que aplica o prefixo de indentação antes de passar
        a mensagem para o logger real.
        """
        prefix = ""
        # Lógica para os estilos de prefixo em cada nível:
        # Nível 1: Um simples marcador de item.
        # Nível 2+: Uma estrutura aninhada com um caractere de "galho".
        if self.indent_level == 1:
            prefix = "╰> "
        elif self.indent_level > 1:
            padding = self.indent_char * (self.indent_level - 2)
            prefix = f"{padding}   ╰─> " # ↳

        self.logger.log(level, f"{prefix}{msg}", *args, **kwargs)

    # --- Métodos Públicos de Logging ---

    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    # --- Métodos de Controle de Indentação ---

    def indent(self, level: int = 1):
        """
        Aumenta o nível de indentação do logger em N níveis.

        :param level: O número de níveis para aumentar. Padrão é 1.
        """
        if level > 0:
            self.indent_level += level

    def dedent(self, level: int = 1):
        """
        Diminui o nível de indentação do logger em N níveis.

        :param level: O número de níveis para diminuir. Padrão é 1.
        """
        if level > 0:
            self.indent_level = max(0, self.indent_level - level)

# ==============================================================================
# BLOCO DE DEMONSTRAÇÃO E TESTE
# ==============================================================================
# O bloco abaixo só é executado quando este arquivo (log_configurator.py)
# é rodado diretamente. Isso o torna um módulo testável de forma independente.

class MockEtlProcess:
    """Uma classe de exemplo para demonstrar o uso do IndentedLogger."""
    def __init__(self):
        self.logger = IndentedLogger(self.__class__.__name__)
        self.logger.info("Processador (Mock) inicializado.")

    def run(self):
        self.logger.info("Iniciando processo de ETL...")
        self.logger.indent()

        self.logger.info("Etapa 1: Leitura de Dados")
        self.logger.indent()
        self.logger.debug("Conectando à fonte de dados...")
        time.sleep(0.1)
        self.logger.info("Leitura de 'tabela_A' concluída.")
        self.logger.info("Leitura de 'tabela_B' concluída.")
        self.logger.dedent()

        self.logger.info("Etapa 2: Transformação de Dados")
        self.logger.indent()
        self.logger.info("Aplicando regras de negócio...")
        time.sleep(0.1)
        self.logger.warning("Encontrados 32 valores nulos na coluna 'X', tratados com a média.")
        self.logger.info("Dados transformados com sucesso.")
        self.logger.dedent()

        self.logger.info("Etapa 3: Escrita de Dados (Simulando Erro)")
        self.logger.indent()
        try:
            self.logger.info("Tentando escrever na tabela de destino...")
            time.sleep(0.1)
            raise ConnectionError("Não foi possível conectar ao banco de dados de destino.")
        except Exception as e:
            self.logger.error(f"Erro durante a escrita de dados. Processo interrompido. Erro: {e}")
        self.logger.dedent()

        self.logger.dedent()
        self.logger.info("Processo de ETL finalizado (com erros).")


if __name__ == '__main__':
    # 1. Configura o ambiente de logging.
    setup_custom_logging(level='DEBUG') # Força o nível DEBUG para ver todas as mensagens

    # 2. Cria um logger principal para o 'orquestrador' do teste.
    main_logger = IndentedLogger('PipelineRunner')

    # 3. Executa a demonstração.
    main_logger.info("Iniciando orquestração do pipeline de teste...")
    main_logger.indent()

    processo_mock = MockEtlProcess()
    processo_mock.run()

    main_logger.dedent()
    main_logger.critical("Orquestração do pipeline de teste finalizada.")