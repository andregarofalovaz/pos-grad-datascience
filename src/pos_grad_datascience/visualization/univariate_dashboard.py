"""
Módulo para renderização do Dashboard de Análise Univariada.

Este módulo contém a classe `UnivariateDashboard`, que consome a classe de
análise `UnivariateAnalysis` para construir e exibir uma interface interativa
completa, utilizando `ipywidgets`.
"""
import ipywidgets as widgets
import pandas as pd
from IPython.display import display, HTML

from pos_grad_datascience.core.log_configurator import IndentedLogger
from pos_grad_datascience.analysis.univariate import UnivariateAnalysis

logger = IndentedLogger(__name__)

class UnivariateDashboard:
    def __init__(self, df: pd.DataFrame):
        self.analyzer = UnivariateAnalysis(df)
        self._column_map = {
            'Análise Numérica': self.analyzer.numeric_cols,
            'Análise Categórica': self.analyzer.categorical_cols
        }
        self._init_ui()

    def _init_ui(self):
        """Inicializa todos os componentes da interface."""
        self._create_selectors()
        self._create_containers()
        self._link_callbacks()

    def _create_selectors(self):
        """Cria os dropdowns de seleção."""
        self.type_dropdown = widgets.Dropdown(options=self._column_map.keys(), description='Tipo de Análise:')
        self.column_dropdown = widgets.Dropdown(options=self._column_map[self.type_dropdown.value], description='Selecionar Coluna:')
        self.selector_box = widgets.HBox([self.type_dropdown, self.column_dropdown], layout=widgets.Layout(grid_gap='20px'))

    def _create_containers(self):
        """Cria o contêiner principal do dashboard."""
        display(HTML("""
        <style>
            .dashboard-container { background-color: #E6E6EE; border: 1px solid #888; padding: 10px; box-sizing: border-box; }
            .inner-panel { background-color: white; border: 1px solid #CCC; border-radius: 8px; padding: 5px; box-sizing: border-box; }
        </style>
        """))
        self.dashboard_output = widgets.Output()
        self.app_container = widgets.VBox([self.selector_box, self.dashboard_output], layout=widgets.Layout(align_items='center'))

    def _link_callbacks(self):
        """Vincula os eventos de mudança dos widgets."""
        self.type_dropdown.observe(self._on_type_change, names='value')
        self.column_dropdown.observe(self._on_column_change, names='value')

    def _on_type_change(self, change):
        """Atualiza as opções do dropdown de colunas."""
        self.column_dropdown.options = self._column_map[change['new']]

    def _on_column_change(self, change):
        """Renderiza o dashboard apropriado (numérico ou categórico)."""
        selected_type = self.type_dropdown.value
        selected_column = change['new']
        
        self.dashboard_output.clear_output(wait=True)
        with self.dashboard_output:
            if not selected_column:
                return # Evita erro se a lista de colunas ficar vazia
                
            logger.info(f"Gerando '{selected_type}' para a coluna: '{selected_column}'")
            if selected_type == 'Análise Numérica':
                self._display_numeric_dashboard(selected_column)
            else:
                self._display_categorical_dashboard(selected_column)

    def _display_numeric_dashboard(self, column):
        """Constrói e exibe o dashboard numérico."""
        # Criação dos painéis de saída
        hist_out = widgets.Output()
        stats_out = widgets.Output()
        box_out = widgets.Output()
        recs_out = widgets.Output()
        
        for panel in [hist_out, stats_out, box_out, recs_out]:
            panel.add_class('inner-panel')

        # --- ALTERAÇÃO: Usando dicionários para criar os layouts de forma segura ---
        top_row_style = {'height': '316px'}
        bottom_row_style = {'height': '164px'}

        hist_out.layout = widgets.Layout(**top_row_style)
        stats_out.layout = widgets.Layout(
            **top_row_style, 
            display='flex', 
            flex_flow='column', 
            justify_content='center', 
            align_items='center'
        )
        box_out.layout = widgets.Layout(**bottom_row_style)
        recs_out.layout = widgets.Layout(**bottom_row_style)
        
        # Montagem do Layout
        left = widgets.VBox([hist_out, box_out], layout=widgets.Layout(width='570px', grid_gap='10px'))
        right = widgets.VBox([stats_out, recs_out], layout=widgets.Layout(width='380px', grid_gap='10px'))
        dashboard = widgets.HBox([left, right], layout=widgets.Layout(width='980px', height='520px', grid_gap='10px'))
        dashboard.add_class('dashboard-container')
        
        # Preenchimento com dados
        with stats_out: display(self.analyzer.get_numeric_stats(column))
        with hist_out: display(self.analyzer.plot_distribution(column))
        with box_out: display(self.analyzer.plot_boxplot(column))
        with recs_out: 
            recs = self.analyzer.get_numeric_recommendations(column)
            if recs:
                html_recs = (recs.replace('**', '').replace('•', '&#8226;').replace('\n', '<br>'))
                html_value = f"<div style='font-size: 11px; line-height: 1.6; padding: 10px; word-wrap: break-word;'>{html_recs}</div>"
                display(HTML(html_value))
        
        display(dashboard)

    def _display_categorical_dashboard(self, column):
        """Constrói, preenche e exibe o dashboard categórico."""
        # 1. Cria os painéis de saída
        bar_out = widgets.Output(); freq_out = widgets.Output()
        rare_out = widgets.Output(); recs_out = widgets.Output()
        
        # 2. Monta o layout (aqui podemos futuramente customizar)
        dashboard = self._build_categorical_layout(bar_out, freq_out, rare_out, recs_out)
        
        # 3. Preenche os painéis com os dados da análise
        with bar_out: display(self.analyzer.plot_categorical_distribution(column))
        with freq_out: display(self.analyzer.get_categorical_stats(column))
        with rare_out:
            title = widgets.HTML("<h6>Categorias Raras (< 1%):</h6>"); display(title)
            rare_table = self.analyzer.get_rare_categories(column)
            if rare_table is not None: display(rare_table)
            else: display(HTML("<i style='color:#666;'>Nenhuma encontrada.</i>"))
        with recs_out:
            recs = self.analyzer.get_categorical_recommendations(column)
            if recs:
                html_recs = recs.replace('•', '&#8226;').replace('\n', '<br>')
                display(HTML(f"<div style='font-size: 11px;...'>{html_recs}</div>"))

        # 4. Exibe o dashboard montado e preenchido
        display(dashboard)


    def _build_categorical_layout(self, bar_out, freq_out, rare_out, recs_out):
        """Função auxiliar que apenas constrói a estrutura visual do dashboard categórico."""
        for panel in [bar_out, freq_out, rare_out, recs_out]: panel.add_class('inner-panel')
        
        top_layout = widgets.Layout(height='316px')
        bottom_layout = widgets.Layout(height='164px')

        bar_out.layout = top_layout; freq_out.layout = top_layout
        rare_out.layout = bottom_layout; recs_out.layout = bottom_layout

        left = widgets.VBox([bar_out, rare_out], layout=widgets.Layout(width='570px', grid_gap='10px'))
        right = widgets.VBox([freq_out, recs_out], layout=widgets.Layout(width='380px', grid_gap='10px'))
        
        dashboard = widgets.HBox([left, right], layout=widgets.Layout(width='980px', height='520px', grid_gap='10px'))
        dashboard.add_class('dashboard-container')
        
        return dashboard
    
    def display(self):
        """Renderiza o dashboard completo e aciona a primeira análise."""
        display(self.app_container)
        self._on_column_change({'new': self.column_dropdown.value})