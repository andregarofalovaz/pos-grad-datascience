import pandas as pd
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from IPython.display import display, HTML
import io 

class UnivariateDashboard:
    """
    Uma classe para gerar um dashboard interativo de análise univariada
    para um DataFrame do pandas dentro de um ambiente Jupyter.

    Uso:
        dashboard = UnivariateDashboard(meu_dataframe)
        dashboard.display()
    """
    # Bloco de CSS movido para uma constante de classe para melhor organização.
    _DASHBOARD_STYLE = """
    <style>
        .widget-label { color: black !important; font-weight: 500; }
        .dashboard-container { background-color: #E6E6EE; border: 1px solid #888; border-radius: 8px; padding: 10px; box-sizing: border-box; display: flex; align-items: stretch; }
        .inner-panel { background-color: white; border: 1px solid #CCC; border-radius: 8px; box-sizing: border-box; overflow: auto; }
        .selector-box { margin-bottom: 15px; padding: 10px; background-color: #F7F7FF; border: 1px solid #DDD; border-radius: 8px; box-sizing: border-box; }
        .custom-stats-table { width: 95%; margin: 10px auto; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px; border: 1px solid #dcdcdc; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .custom-stats-table caption { font-size: 1.1em; font-weight: 600; color: #333; margin: 8px 0; text-align: left; padding-left: 12px; }
        .custom-stats-table th, .custom-stats-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; color: #333; }
        .custom-stats-table tr:last-child td { border-bottom: none; }
        .custom-stats-table td:last-child { text-align: right; font-weight: 500; }
        .categorical-stats-table { width: 95%; margin: 10px auto; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; }
        .categorical-stats-table th, .categorical-stats-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #e0e0e0; color: #333; }
        .categorical-stats-table th { font-weight: 600; }
    </style>
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa o dashboard com um DataFrame.

        Args:
            data (pd.DataFrame): O DataFrame a ser analisado.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("O dado de entrada deve ser um DataFrame pandas não vazio.")

        self.data = data.copy() # Usar .copy() para evitar efeitos colaterais no df original

        # Lógica de preparação de dados (Model) movida para o construtor.
        self.numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = self.data.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        self.column_map = {
            'Numérica': self.numeric_cols,
            'Categórica': self.categorical_cols,
            'Data': self.date_cols
        }

        # Atributos de instância para widgets e layout inicializados como None.
        self.type_dropdown = None
        self.column_dropdown = None
        self.content_panels = None
        self.dashboard_container = None
        self.layout_config = None
        self.app_container = None

    def _create_layout(self):
        """Cria e configura os componentes de layout do dashboard."""
        display(HTML(self._DASHBOARD_STYLE))

        self.layout_config = {
            'dashboard_width': '980px', 'dashboard_height': '500px', 'gap': '10px',
            'left_panel_width': '570px', 'right_panel_width': '380px'
        }
        cfg = self.layout_config
        self.content_panels = {'left': widgets.Output(), 'right': widgets.Output()}

        for panel in self.content_panels.values():
            panel.add_class('inner-panel')

        self.content_panels['left'].layout = widgets.Layout(width=cfg['left_panel_width'])
        self.content_panels['right'].layout = widgets.Layout(width=cfg['right_panel_width'])

        self.dashboard_container = widgets.HBox(
            list(self.content_panels.values()),
            layout=widgets.Layout(width=cfg['dashboard_width'], height=cfg['dashboard_height'], grid_gap=cfg['gap'])
        )
        self.dashboard_container.add_class('dashboard-container')

    def _on_type_change(self, change):
        """Callback para atualizar as opções de coluna quando o tipo de análise muda."""
        if self.column_dropdown:
            self.column_dropdown.options = self.column_map.get(change['new'], [])

    # --- Métodos Privados para Geração de Conteúdo ---

    def _display_numeric_stats(self, column: str):
        stats = self.data[column].describe()
        stats['skew'] = self.data[column].skew()
        stats['nulls'] = self.data[column].isnull().sum()
        stats_df = stats.to_frame().reset_index()
        stats_df.columns = ['Métrica', 'Valor']

        metric_map = {
            'count': 'Quantidade', 'mean': 'Média', 'std': 'Desvio Padrão', 'min': 'Mínimo',
            '25%': '1º Quartil (Q1)', '50%': 'Mediana (Q2)', '75%': '3º Quartil (Q3)',
            'max': 'Máximo', 'skew': 'Assimetria', 'nulls': 'Valores Nulos'
        }
        html = f"""<table class="custom-stats-table"><caption>Estatísticas: {column}</caption><tbody>"""
        for _, row in stats_df.iterrows():
            metric_name = metric_map.get(row['Métrica'], row['Métrica'])
            value = f"{int(row['Valor']):,}" if row['Métrica'] in ['count', 'nulls'] else f"{row['Valor']:,.2f}"
            html += f"""<tr><td>{metric_name}</td><td>{value}</td></tr>"""
        html += """</tbody></table>"""
        display(HTML(html))

    def _plot_numeric(self, column: str):
        """Renderiza o gráfico numérico em memória e retorna como um widget de Imagem."""
        sns.set_style("whitegrid")

        # 1. Cria a figura e os eixos com um figsize de referência e mais compacto
        fig, axes = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(6.8, 4.5), # Tamanho base ajustado
            gridspec_kw={'height_ratios': [0.3, 0.7]}
        )
        fig.suptitle(f'Distribuição de {column}', fontsize=14)

        # 2. Lógica de plotagem (permanece a mesma)
        # Boxplot
        sns.boxplot(x=self.data[column], ax=axes[0], color='skyblue')
        axes[0].set_title('Boxplot e Quartis', fontsize=10)
        axes[0].set_xlabel(''); axes[0].set_yticks([])
        quantiles = self.data[column].quantile([0.25, 0.5, 0.75]).values
        for val in quantiles:
            axes[0].text(val, 0.0, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        # Histograma
        sns.histplot(self.data[column], kde=True, bins=30, ax=axes[1])
        axes[1].set_title('Histograma de Frequência', fontsize=10)
        axes[1].set_xlabel(column, fontsize=9)
        axes[1].set_ylabel('Frequência', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajuste para dar espaço ao suptitle

        # 3. Processo de renderização em memória (A MUDANÇA PRINCIPAL)
        buf = io.BytesIO()
        # Salva a figura no buffer, cortando todas as margens extras
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        buf.seek(0)

        # Cria um widget de imagem com os dados do buffer
        image_widget = widgets.Image(value=buf.read(), format='png')

        # Limpa a figura da memória para evitar que ela seja exibida duas vezes
        plt.close(fig)

        # 4. Retorna o widget de imagem em vez de chamar plt.show()
        return image_widget

    def _display_categorical_stats(self, column: str):
        counts = self.data[column].value_counts()
        percentages = self.data[column].value_counts(normalize=True) * 100
        freq_df = pd.DataFrame({'Categoria': counts.index, 'Contagem': counts.values, 'Percentual': percentages.values})

        html = """<table class="categorical-stats-table"><thead><tr><th>Categoria</th><th>Contagem</th><th>Percentual</th></tr></thead><tbody>"""
        for _, row in freq_df.head(10).iterrows(): # Limita a 10 para não sobrecarregar
             html += f"<tr><td>{row['Categoria']}</td><td>{row['Contagem']:,}</td><td>{row['Percentual']:.2f}%</td></tr>"
        html += """</tbody></table>"""
        display(HTML(html))


    def _plot_categorical(self, column: str):
        """Renderiza o gráfico categórico em memória e retorna como um widget de Imagem."""
        sns.set_style("whitegrid")
        
        # 1. Cria a figura com um figsize de referência (ainda é importante)
        fig = plt.figure(figsize=(6.8, 4.2))
        ax = fig.add_subplot(1, 1, 1)

        # 2. Lógica de plotagem
        counts = self.data[column].value_counts()
        top_n = 10
        if len(counts) > top_n:
            top_data = counts.nlargest(top_n)
            others_sum = counts.nsmallest(len(counts) - top_n).sum()
            plot_data = pd.concat([top_data, pd.Series({'Outros': others_sum})])
        else:
            plot_data = counts

        sns.barplot(x=plot_data.index, y=plot_data.values, hue=plot_data.index, palette="viridis_r", legend=False, ax=ax)
        ax.set_title(f'Contagem de Categorias para {column}', fontsize=14, pad=15)
        ax.set_xlabel('Categorias', fontsize=11)
        ax.set_ylabel('Contagem', fontsize=11)
        # for i, v in enumerate(plot_data.values):
        #     ax.text(i, v + (plot_data.values.max() * 0.01), f'{int(v):,}', color='black', ha='center', fontsize=9)
        plt.xticks(rotation=40, ha='right')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        # 3. Processo de renderização em memória
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        buf.seek(0)
        image_widget = widgets.Image(value=buf.read(), format='png')
        plt.close(fig)
        return image_widget


    def _plot_date(self, column: str):
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4))
        counts_per_month = self.data.set_index(column).resample('M').size()
        counts_per_month.plot(kind='line', marker='o', linestyle='-')
        plt.title(f'Ocorrências Mensais: {column}', fontsize=14)
        plt.xlabel('Mês', fontsize=12)
        plt.ylabel('Número de Ocorrências', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def _update_dashboard(self, change):
        """Callback principal que despacha a atualização dos painéis."""
        selected_column = change['new']
        if not selected_column: return

        # Dispatcher refatorado para usar dicionários, melhorando a escalabilidade.
        plot_handlers = {
            'Numérica': self._plot_numeric,
            'Categórica': self._plot_categorical,
            'Data': self._plot_date
        }
        stats_handlers = {
            'Numérica': self._display_numeric_stats,
            'Categórica': self._display_categorical_stats
        }
        
        selected_type = self.type_dropdown.value
        
        # Atualiza o painel esquerdo (gráficos)
        self.content_panels['left'].clear_output(wait=True)
        with self.content_panels['left']:
            if handler := plot_handlers.get(selected_type):
                # ALTERAÇÃO: Captura o widget retornado e o exibe
                plot_widget = handler(selected_column)
                display(plot_widget)

        # Atualiza o painel direito (estatísticas)
        self.content_panels['right'].clear_output(wait=True)
        with self.content_panels['right']:
            if handler := stats_handlers.get(selected_type):
                handler(selected_column)
            else:
                 display(HTML(f"<i>Análise complementar para <b>{selected_column}</b>.</i>"))

    def display(self):
        """Monta e exibe o dashboard interativo completo."""
        self._create_layout()

        # Criação dos widgets de controle (Controller)
        self.type_dropdown = widgets.Dropdown(
            options=self.column_map.keys(), description='Tipo de Análise:', style={'description_width': 'initial'}
        )
        self.column_dropdown = widgets.Dropdown(
            options=self.column_map[self.type_dropdown.value], description='Selecionar Coluna:', style={'description_width': 'initial'}
        )

        # Vinculação dos callbacks
        self.type_dropdown.observe(self._on_type_change, names='value')
        self.column_dropdown.observe(self._update_dashboard, names='value')

        # Montagem final da UI
        selector_box = widgets.HBox(
            [self.type_dropdown, self.column_dropdown],
            layout=widgets.Layout(width=self.layout_config['dashboard_width'], grid_gap='20px', justify_content='flex-start')
        )
        selector_box.add_class('selector-box')

        self.app_container = widgets.VBox(
            [selector_box, self.dashboard_container], layout=widgets.Layout(align_items='center')
        )

        display(self.app_container)
        
        # Dispara a primeira renderização
        if self.column_dropdown.value:
            self._update_dashboard({'new': self.column_dropdown.value})