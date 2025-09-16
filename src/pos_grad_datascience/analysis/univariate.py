"""
Módulo para Análise Univariada de Dados.

Este módulo fornece a classe `UnivariateAnalysis`, uma suíte de ferramentas
projetada para realizar a Etapa 3-A (Análise Univariada) do pipeline de
análise de dados. A classe oferece métodos para gerar estatísticas descritivas,
visualizar distribuições e obter recomendações de pré-processamento de forma
sistemática para variáveis numéricas e categóricas.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from typing import Optional

from pos_grad_datascience.core.log_configurator import IndentedLogger
from pos_grad_datascience.core.decorators import validate_column_type

logger = IndentedLogger(__name__)

# Dicionário para tradução de métricas estatísticas para português.
METRIC_TRANSLATION = {
    'count': 'Contagem',
    'mean': 'Média',
    'std': 'Desvio Padrão',
    'min': 'Mínimo',
    '25%': '1º Quartil (Q1)',
    '50%': 'Mediana (Q2)',
    '75%': '3º Quartil (Q3)',
    'max': 'Máximo'
}

class UnivariateAnalysis:
    """
    Encapsula funcionalidades para a análise univariada de um DataFrame.

    Ao ser instanciada, a classe separa automaticamente as colunas numéricas e
    categóricas do DataFrame fornecido, disponibilizando métodos específicos
    para a análise de cada tipo de variável.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa a classe de análise univariada.

        Args:
            df (pd.DataFrame): O DataFrame a ser analisado.

        Raises:
            TypeError: Se o objeto de entrada não for um pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("O objeto de entrada deve ser um pandas DataFrame.")
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()


    @validate_column_type('numeric')
    def get_numeric_stats(self, column_name: str) -> Optional[Styler]:
        """Calcula e formata estatísticas descritivas para uma coluna numérica.

        Args:
            column_name (str): O nome da coluna numérica a ser analisada.

        Returns:
            Optional[Styler]: Um objeto Styler do Pandas com a tabela de
                              estatísticas formatada, ou None em caso de erro.
        """
        if column_name not in self.numeric_cols:
            logger.error("Coluna '%s' não é uma coluna numérica válida.", column_name)
            return None
        try:
            stats_series = self.df[column_name].describe()
            
            # Adiciona a métrica de percentual da moda
            mode_value = self.df[column_name].mode().iloc[0]
            mode_perc = (self.df[column_name] == mode_value).sum() / len(self.df) * 100
            stats_series[f'% da Moda ({mode_value})'] = mode_perc

            # Formatação do relatório
            stats_df = stats_series.to_frame(name='Valor').reset_index()
            stats_df.rename(columns={'index': 'Métrica_EN'}, inplace=True)
            stats_df['Métrica'] = stats_df['Métrica_EN'].apply(
                lambda x: METRIC_TRANSLATION.get(x, x.replace("perc_mode", "% da Moda"))
            )
            
            return (stats_df[['Métrica', 'Valor']]
                    .style.format({'Valor': '{:.4f}'})
                    .hide(axis="index"))

        except Exception as e:
            logger.error("Falha ao calcular estatísticas para '%s': %s", column_name, e)
            return None
        
    def plot_distribution(self, column_name: str) -> Optional[go.FigureWidget]:
        """Plota um histograma interativo e uma curva de densidade (KDE).

        Args:
            column_name (str): O nome da coluna numérica a ser plotada.

        Returns:
            Optional[go.FigureWidget]: Um objeto FigureWidget do Plotly ou None.
        """
        if column_name not in self.numeric_cols: return None
        try:
            fig = px.histogram(self.df, x=column_name, marginal="rug", histnorm='probability density')
            fig.update_layout(
                title_text=f'Distribuição de "{column_name}"',
                yaxis_title='Densidade',
                xaxis_title=None,
                template='plotly_white',
                margin=dict(t=50, b=10, l=10, r=10),
                height=304,
                width=650
            )
            return go.FigureWidget(fig)
        except Exception as e:
            logger.error("Falha ao plotar distribuição para '%s': %s", column_name, e)
            return None

    def plot_boxplot(self, column_name: str) -> Optional[go.FigureWidget]:
        """Plota um boxplot interativo para análise de dispersão e outliers.

        Args:
            column_name (str): O nome da coluna numérica a ser plotada.

        Returns:
            Optional[go.FigureWidget]: Um objeto FigureWidget do Plotly ou None.
        """
        if column_name not in self.numeric_cols: return None
        try:
            fig = px.box(self.df, x=column_name)
            fig.update_layout(
                title_text=f'Boxplot de "{column_name}"',
                template='plotly_white',
                margin=dict(t=50, b=10, l=10, r=10),
                height=152,
                width=650
            )
            return go.FigureWidget(fig)
        except Exception as e:
            logger.error("Falha ao plotar boxplot para '%s': %s", column_name, e)
            return None
        
    def get_numeric_recommendations(self, column_name: str) -> Optional[str]:
        """Gera um resumo textual com recomendações de pré-processamento.

        Args:
            column_name (str): O nome da coluna numérica.

        Returns:
            Optional[str]: Uma string formatada com as recomendações, ou None.
        """
        if column_name not in self.numeric_cols:
            logger.error("Coluna '%s' não é numérica.", column_name)
            return None
        
        recommendations = ["**Indicações de Pré-processamento:**"]
        col_data = self.df[column_name]
        
        # Análise de Assimetria
        skewness = col_data.skew()
        if abs(skewness) > 1.0:
            recommendations.append(f"• **Assimetria Alta ({skewness:.2f}):** Considere transformações (Log, Box-Cox).")
        elif abs(skewness) > 0.5:
            recommendations.append(f"• **Assimetria Moderada ({skewness:.2f}):** Normalização pode ser benéfica.")
            
        # Análise de Outliers
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        outliers_count = ((col_data < (q1 - 1.5 * (q3 - q1))) | (col_data > (q3 + 1.5 * (q3 - q1)))).sum()
        if outliers_count > 0:
            recommendations.append(f"• **Outliers Detectados ({outliers_count}):** Considere usar `RobustScaler`.")

        # Análise de Escala
        if col_data.std() > 1000 or col_data.max() > 1000:
             recommendations.append("• **Escala Grande:** Padronização (`StandardScaler`) é essencial.")

        # Análise de Zero-Inflação
        if (col_data == 0).mean() > 0.8:
            recommendations.append(f"• **Zero-Inflada ({(col_data == 0).mean()*100:.1f}% de zeros):** Considere engenharia de atributos.")
            
        if len(recommendations) == 1:
            recommendations.append("• A distribuição parece bem comportada. `StandardScaler` é um bom ponto de partida.")
            
        return "\n".join(recommendations)

    @validate_column_type('categorical')
    def get_categorical_stats(self, column_name: str) -> Optional[go.FigureWidget]:
        """Cria uma tabela de frequência interativa para uma coluna categórica."""
        if column_name not in self.categorical_cols: return None
        try:
            counts = self.df[column_name].value_counts(dropna=False)
            freq_table = pd.DataFrame({
                'Categoria': counts.index.astype('object').fillna('Valores Nulos'),
                'Contagem': counts.values,
                'Percentual (%)': (counts.values / len(self.df) * 100)
            })
            
            # Lógica para destacar a fonte de valores nulos
            font_colors = [['red' if cat == 'Valores Nulos' else 'black' for cat in freq_table['Categoria']]] * len(freq_table.columns)

            fig = go.Figure(data=[go.Table(
                header=dict(values=list(freq_table.columns), fill_color='#4C9181', font=dict(color='white')),
                cells=dict(values=freq_table.transpose().values.tolist(), format=[None, ",.0f", ".2f%"], font=dict(color=font_colors))
            )])
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            return go.FigureWidget(fig)
        except Exception as e:
            logger.error(f"Falha ao gerar tabela de stats para '{column_name}': {e}")
            return None

    def plot_categorical_distribution(self, column_name: str, top_n: int = 15) -> Optional[go.FigureWidget]:
        """Cria um gráfico de barras interativo da distribuição categórica."""
        if column_name not in self.categorical_cols: return None
        try:
            freq = self.df[column_name].value_counts(dropna=False).nlargest(top_n).reset_index()
            freq.columns = ['Categoria', 'Contagem']
            freq['Categoria'] = freq['Categoria'].astype('object').fillna('Valores Nulos')
            
            colors = ['red' if cat == 'Valores Nulos' else '#4C9281' for cat in freq['Categoria']]
            
            fig = px.bar(freq, x='Categoria', y='Contagem', text_auto=True, 
                         title=f'Distribuição de "{column_name}" (Top {top_n})')
            fig.update_layout(template='plotly_white', xaxis_title=None, height=300)
            fig.update_traces(marker_color=colors)
            return go.FigureWidget(fig)
        except Exception as e:
            logger.error(f"Falha ao gerar gráfico de distribuição para '{column_name}': {e}")
            return None