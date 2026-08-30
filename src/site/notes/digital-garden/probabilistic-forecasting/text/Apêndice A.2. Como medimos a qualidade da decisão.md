---
{"dg-publish":true,"permalink":"/digital-garden/probabilistic-forecasting/text/apendice-a-2-como-medimos-a-qualidade-da-decisao/"}
---

# Apêndice A.2 — Como medimos a qualidade da decisão?

Antes de escrever qualquer linha de código ou escolher uma arquitetura — XGBoost, redes neurais, regressão logística, *conformal prediction* ou um *solver* de otimização — estruture o problema em sua complexidade real.

A premissa central é: separe a decisão — a alavanca sob seu controle — do resultado, que acontece depois sob influência do ambiente e da aleatoriedade.

---

Este Framework não é novidade alguma. Ela é apenas o Framework já conhecido de Warren Powell para otimização de sistemas, apenas com uma vestimenta diferente observada para projetos de ciência de dados. Ao contrário de outros frameworks, acredito que o dele resolve uma dor que sinto em alguns frameworks projetados em Machine Learning: focam na solução, não no negócio. Acredito que se tivermos um design, um passo antes de tudo, pensando no problema como negócio e o que rege ele, o modelo passa a traduzir melhor para diretores e C-level.

Outro ponto importante a comentar, é que estruturei este texto propositalmente para ser utilizado junto a qualquer LLM, para ele ser um assistente à discussão.

---
## Passo 2: Como Medimos a Qualidade da Decisão? (O Triângulo O - M - C)
---
A base do framework assume que uma decisão é qualquer alavanca que você controla (ex: aprovar/recusar crédito, definir o preço, escolher os 3 itens do ranking, enviar ou não um cupom). É essencial não confundir a decisão que você toma com o resultado estocástico posterior.

Segundo o livro [The Decision Factory](https://www.amazon.com.br/Decision-Factory-Novel-Decisions-Uncertainty/dp/B0GN8SJXR1), muitos times cometem o erro de misturar regras duras, métricas estatísticas e metas financeiras dentro de uma única função com pesos arbitrários ($\lambda$), ocultando os _trade-offs_ reais do negócio. O framework exige a separação rigorosa entre três entidades:

### 1. Objetivo
---
É a única função matemática que o algoritmo ou política de decisão tentará maximizar ou minimizar de forma estrita. Ela traduz a meta econômica final.

> **O Teste da Bússola Única:** _Qual é a quantidade financeira ou operacional exata que define, numericamente, se a operação foi rentável ou bem-sucedida?_

- **Crédito e Prevenção à Fraude:**
$$\max \mathbb{E}[\text{Lucro}] = (\text{Receita de Juros e Tarifas}) - (\text{Custo de Inadimplência Previsto} \times \hat{P}(\text{Default})) - (\text{Custo de Atrito com Falsos Positivos})$$
- **Sistemas de Recomendação / E-commerce:**
$$\max \mathbb{E}[\text{GMV}] = \sum_{i \in \text{ranking}} \text{Preço}_i \times \text{Margem}_i \times \hat{P}(\text{Clique}_i) \times \hat{P}(\text{Compra}_i \mid \text{Clique}_i)$$
#### Função de Perda vs. Função Objetivo
---
- **A Função de Perda ($\mathcal{L}(\hat{y}, y)$):** É o critério  usado durante o treinamento do algoritmo (ex: _Log-Loss_, _Mean Squared Error_, _Focal Loss_). Ela serve apenas para calibrar crenças estatísticas ($B_t$). **A Loss nunca é o objetivo do negócio.**
    
- **O Objetivo da Decisão ($\max \mathbb{E}[U(a \mid S_t)]$):** É a função que avalia o impacto econômico e operacional da ação tomada $a_t$. O modelo preditivo apenas fornece insumos de probabilidade para que a decisão maximize essa utilidade real.

### 2. Restrição
---
É um limite rígido, absoluto e não negociável (seja ele físico, legal, computacional ou de negócio). Se uma restrição for violada, a solução é **inválida** e o sistema não pode executar a ação. Otimizadores e políticas de decisão sequer devem avaliar cenários fora deste espaço viável.

> **O Teste da Linha Vermelha:** _Se o modelo sugerir uma ação que ultrapasse esse limite, o sistema quebra, a empresa toma um processo regulatório ou a infraestrutura cai?_ Se a resposta for sim, é uma restrição rígida ($C$), e não uma métrica.

**Exemplos:**

- **Restrições Físicas e de Interface (Espaço em Tela):** O carrossel de recomendação só possui $k=3$ posições visíveis; não é possível recomendar 4 itens.
    
- **Restrições Legais e Regulatórias (_Compliance_):** O modelo de crédito é estritamente proibido por lei (BACEN/LGPD) de usar atributos protegidos (raça, gênero, orientação) na tomada de decisão de concessão.
    
- **Restrições Computacionais e de Engenharia (SLA Técnico):** O tempo total de inferência do pipeline não pode ultrapassar $80\text{ ms}$ (limite rígido para não causar _timeout_ no checkout).
    
- **Restrições Orçamentárias Rígidas:** O montante total de crédito concedido ou de cupons distribuídos na campanha no dia não pode ultrapassar o limite financeiro alocado ($R_t \ge 0$).
    
- **Restrições Lógicas de Estoque:** É proibido recomendar ou vender SKUs cujo estoque físico em $R_t$ seja igual a zero.
		

### 3. Métrica
---
São os indicadores que os humanos (stakeholders de negócio, cientistas de dados e executivos) acompanham no painel de controle para avaliar a saúde, estabilidade, justiça e qualidade do sistema ao longo do tempo.

> **Atenção à Armadilha Clássica:** Não transforme automaticamente uma meta de negócio ou KPI em restrição matemática. Primeiro explicite o *trade-off* que ele representa. Porém, se sua violação tornar a ação ilegal, contratualmente inadmissível ou operacionalmente impossível, ele deixa de ser apenas uma métrica e passa a ser uma restrição.
	
**Exemplos de Métricas a Monitorar:**

|**Categoria**|**O que Mede no Sistema**|**Exemplos Práticos em Projetos de ML**|
|---|---|---|
|**Métricas de Negócio**|O impacto financeiro e operacional final.|Taxa de Conversão, Taxa de Aprovação de Crédito, _Customer Lifetime Value_ (LTV), Volume Bruto de Mercadorias (GMV), Taxa de Cancelamento (_Churn_).|
|**Métricas Estatísticas / ML**|A qualidade e calibração das crenças ($B_t$).|AUC-ROC, Log-Loss, Precisão no Top 1%, Erro de Calibração Esperado (ECE), Cobertura Marginal em _Conformal Prediction_.|
|**Métricas de Equidade (_Fairness_)**|A ausência de viés desproporcional entre grupos.|Paridade Demográfica (_Demographic Parity_), Igualdade de Oportunidade (_Equal Opportunity_), Disparate Impact Ratio.|
|**Métricas Operacionais / MLOps**|A estabilidade do modelo e da infraestrutura.|Latência p95/p99, _Data Drift_ (PSI/KS-test), Volume de requisições por segundo (RPS), Taxa de Falsos Positivos.|
|**Estabilidade da Decisão**|O grau de variação das decisões no tempo.|Quão pouco o limite de crédito ou a recomendação de um mesmo usuário oscila de um dia para o outro sem fatos novos.|

>[!tip] Objetivo vs Métrica
>**O Objetivo** é o que o algoritmo tenta otimizar de forma estrita no papel (ex: o número exato em reais do custo total que queremos minimizar). É a bússola matemática da decisão.
>
>**A Métrica** é o painel de controle (KPI) que os humanos usam para avaliar o sucesso real do negócio depois que a decisão é tomada, muitas vezes olhando para fatias específicas da operação (como o nível de serviço de uma região específica ou o atraso médio).
---

Muitas vezes usamos uma métrica (como o Nível de Serviço) para julgar se o Objetivo matemático foi bom o suficiente no mundo real. Por exemplo: o algoritmo minimizou o _custo financeiro_ (Objetivo), mas a diretoria vai olhar para o _nível de serviço de 95%_ (Métrica) para ver se o custo foi reduzido sacrificando demais a qualidade da entrega.

## Objetivo vs. Restrição vs. Métrica em Ação
---
Para consolidar o alinhamento entre a equipe técnica e a diretoria, o papel de cada vértice do triângulo pode ser resumido da seguinte forma:

- **O Objetivo** é a **bússola matemática** que o algoritmo persegue de forma cega no papel (ex: maximizar o lucro líquido esperado).
    
- **A Restrição** é o **muro intransponível** que delimita onde o algoritmo pode operar (ex: respeitar o SLA de $80\text{ ms}$, o orçamento máximo de descontos e as leis do setor).
    
- **A Métrica** é o **painel de instrumentos** que a diretoria utiliza para avaliar se a otimização matemática não está sacrificando aspectos vitais do negócio.
    
**Exemplo Prático de Negociação de Trade-off:**

> O algoritmo minimizou com sucesso o custo de risco de crédito (Objetivo atingido), mas a diretoria analisa a taxa de aprovação comercial (Métrica de 40%) e percebe que o modelo foi excessivamente conservador, travando o crescimento da empresa. Com base nessa métrica, a liderança ajusta o apetite de risco da política de decisão, reequilibrando a fronteira de eficiência entre risco e volume.

## Triângulo O-M-C
---
```
                         [ OBJETIVO (O) ]
                     A Bússola Matemática Única
                     (Impacto Econômico / Utilidade)
                               ▲
                              / \
                             /   \
                            /     \
    [ RESTRIÇÕES (C) ] ◄───'       '───► [ MÉTRICAS (M) ]
    Limites Rígidos Invioláveis         Painel Multidimensional (KPIs)
    (Físicos, Legais, SLAs, Lógica)     (Estatística, Negócio, Equidade)

```

| **Vértice**        | **Definição Operacional**    | **Papel no Sistema**                                                              | **Pergunta-Chave de Validação**                                                | **Exemplo Prático (ML / DS)**                                                                              |
| ------------------ | ---------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Objetivo (O)**   | Bússola Matemática Única     | Maximizar ou minimizar estritamente o valor econômico/utilidade da decisão.       | _Qual valor monetário ou operacional único define o sucesso da decisão?_       | Maximizar o Lucro Líquido Esperado (Receita menos Perdas por Inadimplência e Churn).                       |
| **Restrições (C)** | Linhas Vermelhas Invioláveis | Delimitar o espaço viável; soluções fora daqui quebram a operação ou violam leis. | _Se ultrapassarmos esse limite, a operação quebra ou incorremos em infração?_  | SLA de inferência $\le 80\text{ ms}$, orçamento diário de cupom $\le R\$ 50.000$, conformidade LGPD/BACEN. |
| **Métricas (M)**   | Painel de Controle (KPIs)    | Monitorar _trade-offs_, qualidade estatística, equidade e estabilidade.           | _Como os stakeholders avaliam se a otimização não sacrificou aspectos vitais?_ | Taxa de Aprovação (Negócio), AUC-ROC/ECE (ML), Paridade Demográfica (Fairness), Latência p99 (MLOps).      |

## Roteiro para discussão
---
Use estas perguntas para enquadrar o problema com o time ou orientar uma LLM:

- Qual resultado econômico ou operacional a decisão deve maximizar ou minimizar?
- Quais condições tornam uma ação inviável, independentemente de seu retorno esperado?
- Quais indicadores devem ser monitorados para revelar efeitos indesejados ou *trade-offs*?
- Quais métricas são apenas metas negociáveis e quais, de fato, precisam tornar-se restrições?
- Como a política será revisada se objetivo, restrições e métricas entrarem em conflito?
