# Análise Preditiva de Tempo de Entrega em Delivery

## 1. Contexto do Problema
Este projeto aborda o desafio de prever com precisão o tempo de entrega em um serviço de delivery, um fator essencial para a satisfação do cliente e a eficiência logística. O objetivo é construir um modelo de machine learning a partir de dados operacionais, como informações do entregador, trânsito e características do pedido, para fornecer estimativas de tempo mais confiáveis.


## 2. Metodologia
O desenvolvimento seguiu um pipeline estruturado de ciência de dados nas seguintes etapas:
  1. Configuração do Ambiente: inicialização do projeto, importação de bibliotecas, definição do seed para garantir a reprodutibilidade e configuração da variável de controle para a execução do notebook em modo rápido ou completo;

  2. Análise Inicial do Dataset: um primeiro diagnóstico para entender a estrutura dos dados, verificar a qualidade das colunas, identificar valores ausentes e inconsistências para guiar a etapa de limpeza;
  
  3. Limpeza e Preparação dos Dados: execução de um processo sistemático de limpeza para padronizar nomes de colunas, corrigir tipos de dados, tratar valores nulos e inconsistentes, resultando em um dataset íntegro para a análise;
  
  4. Análise Exploratória dos Dados: investigação visual e estatística para gerar hipóteses, entender a relação entre as variáveis e guiar as decisões de engenharia de atributos;
  
  5. Engenharia de Atributos: criação de variáveis preditivas, como a distância geodésica entre o restaurante e o cliente, o tempo de preparo do pedido e features cíclicas para componentes temporais;
  
  6. Pré-processamento e Modelagem: desenvolvimento de pipeline para processar os dados e alimentar um processo de comparação entre 7 algoritmos distintos. A validação foi realizada com GroupKFold para garantir uma estimativa de performance realista;
  
  7. Interpretação dos Resultados: uso de Feature Importance e SHAP para uma compreensão global e local do comportamento do modelo;
  
  8. Conclusão: consolidação de todos os resultados, com a apresentação da performance final, avaliação de fatores preditivos, limitações do modelo e oportunidades de melhoria.

  
## 3. Resultados
A solução final, baseada em um modelo LightGBM otimizado, alcançou as seguintes métricas de performance no conjunto de teste:

| Métrica | Valor no Teste |
| :--- | :--- |
| **MAE (Erro Médio Absoluto)** | **3.49 minutos** |
| **R² (Coeficiente de Determinação)** | **0.781** |
| **Capacidade de Generalização** | **+0.92%** |

A variação de performance de apenas +0.92% entre a validação cruzada e o conjunto de teste final confirma a capacidade de generalização do modelo.


## 4. Como Executar
#### Requisitos
- Python 3.10+
- Bibliotecas principais: `pandas`, `scikit-learn`, `lightgbm`, `catboost`, `optuna`, `shap`.

#### Instruções
1.  Clone este repositório:
    ```bash
    git clone [https://github.com/andregarofalovaz/pos-grad-datascience.git](https://github.com/andregarofalovaz/pos-grad-datascience.git)
    ```
2.  Abra o notebook `Análise Preditiva de Tempo de Entrega em Delivery.ipynb` em um ambiente Jupyter;

3.  Configure o modo de execução na segunda célula de código:
    - **Modo Rápido (Padrão):** Mantenha a variável `EXECUTAR_MODELAGEM_COMPLETA = False`. O notebook executará em poucos minutos, carregando os resultados pré-calculados da pasta `artifacts/` a partir do GitHub.
    - **Modo Completo:** Altere a variável para `EXECUTAR_MODELAGEM_COMPLETA = True` para executar todo o processo de treinamento e otimização do zero.

