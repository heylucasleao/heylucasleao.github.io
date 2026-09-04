---
{"dg-publish":true,"permalink":"/digital-garden/probabilistic-forecasting/text/apendice-a-1-quais-decisoes-estamos-tomando/","dg-note-properties":{}}
---


# Apêndice A.1 — Quais decisões estamos tomando?

Antes de escrever qualquer linha de código ou escolher uma arquitetura — XGBoost, redes neurais, regressão logística, *conformal prediction* ou um *solver* de otimização — estruture o problema em sua complexidade real.

A premissa central é: separe a decisão — a alavanca sob seu controle — do resultado, que acontece depois sob influência do ambiente e da aleatoriedade.

Este Framework não é novidade alguma. Ela é apenas o Framework já conhecido de Warren Powell para otimização de sistemas, apenas com uma vestimenta diferente observada para projetos de ciência de dados. Ao contrário de outros frameworks, acredito que o dele resolve uma dor que sinto em alguns frameworks projetados em Machine Learning: focam na solução, não no negócio. Acredito que se tivermos um design, um passo antes de tudo, pensando no problema como negócio e o que rege ele, o modelo passa a traduzir melhor para diretores e C-level.

Outro ponto importante a comentar, é que estruturei este texto propositalmente para ser utilizado junto a qualquer LLM, para ele ser um assistente à discussão.

<br>

## Passo 1: Quais decisões estamos tomando? (A Anatomia do Estado)

Em um projeto de ciência de dados, a primeira pergunta é: quem toma a decisão, em qual instante ela ocorre e quais informações estão disponíveis nesse instante? Partir do processo atual ajuda a entender o que pode ser alterado, observado ou automatizado sem perder a ótica de quem hoje detém essa decisão.

Para modelar isso de forma acionável, define-se o **Estado do Sistema ($S_t$)** no tempo $t$. O estado $S_t$ é a fotografia da informação necessária para decidir a ação $a_t$ e prever a evolução do sistema. Ele é decomposto em três tipos fundamentais de informação, além do contexto externo:

<br>

### 1. Estado Físico ($R_t$) — Quantidades e Recursos Fatuais

Representa as quantidades concretas e contáveis registradas no sistema no instante $t$. É uma fotografia factual da operação, não algo que permaneça imutável depois dela. Em ciência de dados, o “físico” abrange recursos tangíveis, capacidades computacionais, limites transacionais e saldos monetários.

> **O Teste da Câmera Fotográfica:** Se eu tirar uma foto da operação no milissegundo $t$, o que consigo contar, medir ou consultar como registro atual? Se depende de uma previsão ou probabilidade, não está em $R_t$.

**Exemplos Práticos estão:**

- **Sistemas de Recomendação / E-commerce:** Quantidade de slots livres no carrossel de produtos ($k=3$), número exato de itens atualmente no carrinho, estoque físico restante do SKU pesquisado no centro de distribuição.

- **Crédito e Prevenção à Fraude:** Saldo disponível em conta corrente no momento da transação, limite de crédito residual utilizável para aquela compra.

- **Plataformas de Mobilidade / Logística:** Número de motoristas logados com status "livre", nível de bateria/combustível de cada veículo, pedidos já aceitos aguardando coleta.

- **Infraestrutura e MLOps:** Orçamento de latência restante para a requisição ($T_{\text{limite}} - T_{\text{decorrido}}$), cota restante de chamadas de API do cliente no dia, número de instâncias de GPU ativas no cluster.

<br>

### 2. Estado de Informação ($I_t$)

Engloba dados cadastrais disponíveis, metadados, regras determinísticas e parâmetros que governam o sistema no instante da decisão. São informações observadas e utilizáveis em $t$, ainda que possam conter problemas de qualidade ou ser atualizadas no futuro; não são previsões sobre o que ocorrerá depois.

> **O Teste da Regra do Jogo:** _Quais são as tabelas de referência, regras duras de negócio e atributos cadastrais que trato como fatos estabelecidos e determinísticos no momento da decisão?_

**Exemplos Práticoss:**

- **Features Cadastrais e Metadados:** Categoria fiscal do lojista, data de criação da conta do usuário, versão ativa do pipeline de engenharia de dados.

- **Histórico Observado (Features de Velocidade):** Número exato de compras confirmadas feitas pelo cartão nos últimos 15 minutos, histórico de transações passadas já liquidadas.

- **Políticas e Regras Duras de Negócio (_Hard Constraints_):** Tabela de tarifas e comissões vigentes, limite total de crédito contratado em contrato, regras regulatórias de bloqueio por lista restritiva (ex: PEP, Sanções, regras LGPD/BACEN).

- **Parâmetros de Calendário e Sistema:** Se o dia atual é feriado nacional ou dia útil, faixa de horário de atendimento humano, taxa de câmbio oficial fixada na abertura do dia.

<br>

### 3. Estado de Crença ($B_t$)

Abriga qualquer estimativa imperfeita, predição, inferência latente ou probabilidade estatística. Em Ciência de Dados, **todo** output gerado por outro modelo, distribuição estatística ou algoritmo de Machine Learning que serve de input para a decisão atual pertence a $B_t$

> **O Teste da Refutabilidade / Inferência Estatística:** _Esse dado é resultado de um cálculo probabilístico, regressão, rede neural ou premissa cujo valor real só poderei confirmar no futuro (ou nunca)?_ (Todo forecast, score e probabilidade mora aqui).

**Exemplos Práticos:**

- **Outputs de Modelos em Cascata / Upstream:**
    - O score de propensão ao cancelamento de um cliente ($\hat{P}(\text{churn}) = 0{,}82$) gerado pelo Modelo A.

    - A probabilidade de uma transação ser fraudulenta calculada por um modelo de _Gradient Boosting_.

    - O vetor de _embeddings_ latentes que estima os gostos e afinidades do usuário.
- **Previsões de Séries Temporais:** Previsão pontual de demanda de energia ou vendas para o horizonte $t+H$ ($\hat{D}_{t, t+H}$).

- **Intervalos e Regiões de Incerteza (_Conformal Prediction_):**
    - O intervalo de predição conforme $[\hat{y}_{\text{inf}}, \hat{y}_{\text{sup}}]$ garantindo 90% de cobertura marginal.

    - O conjunto de classes possíveis retornado por um classificador de conformalidade (ex: $\mathcal{C}(x) = \{\text{Classe A}, \text{Classe C}\}$).
- **Distribuições Posteriores e Bandits:** Parâmetros da distribuição $\text{Beta}(\alpha_t, \beta_t)$ da taxa de conversão de um anúncio em algoritmos de _Thompson Sampling_.

<br>

### 4. Contexto Externo (Features Exógenas)

Informações geradas fora do controle da organização que afetam a operação.

> **O Teste da Fronteira Temporal:** _Quando essa informação se torna um fato observável e registrado para quem toma a decisão?_

| **Categoria Exógena**        | **Onde se Enquadra** | **Critério Temporal**                                                  | **Exemplos Práticos em Ciência de Dados**                                                                                            |
| ---------------------------- | -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Exógeno Fixo**             | $S_0$                | Leis estruturais que não mudam durante o ciclo operacional.            | Leis da física, protocolo TCP/IP, restrições regulatórias do setor financeiro.                                                       |
| **Exógeno Observado**        | $I_t$                | Fatos externos já confirmados e registrados exatamente em $t$.         | A geolocalização do IP de onde partiu a requisição, a temperatura medida pelo sensor em $t$, a taxa Selic/câmbio fixada na abertura. |
| **Exógeno Previsto**         | $B_t$                | Expectativa ou modelo preditivo sobre uma variável externa futura.     | Previsão meteorológica para as próximas 6 horas, estimativa de tráfego na rota, projeção de inflação para o trimestre.               |
| **Choque Exógeno Realizado** | $W_{t+1}$            | Informação estocástica revelada apenas após a tomada da decisão $a_t$. | Ver detalhamento abaixo.                                                                                                             |

<br>

### O Choque Exógeno Realizado ($W_{t+1}$): A Resposta do Ambiente Pós-Decisão

O vetor $W_{t+1}$ representa a informação nova que chega ao sistema logo após a aplicação da decisão $a_t$. Em Ciência de Dados, ele se manifesta em três frentes principais:

1. **O Feedback Imediato do Usuário:**

    - O usuário clicou ou ignorou o banner recomendado na posição 1?

    - O cliente aceitou a proposta de renegociação de dívida ofertada?

2. **O Rótulo Verdadeiro Atrasado (_Delayed Ground Truth / Censored Feedback_):**

    - A transação aprovada em $t$ resultou em contestação (_chargeback_) 45 dias depois ($t+45$).

    - _Problema do Contrafactual Censurado:_ Se a decisão $a_t$ foi "negar crédito", nunca saberemos se o tomador teria honrado a fatura em $t+1$.

3. **A Degradação do Ambiente (_Data Drift / Concept Drift_):**

    - A alteração abrupta no padrão de consumo da população decorrente de um evento macroeconômico, alterando a relação $P(Y \mid X)$ em produção.

<br>

## Roteiro para discussão

Use estas perguntas para enquadrar o problema com o time ou orientar uma LLM:

- Qual ação concreta será escolhida, por quem e em qual frequência?

- Quais recursos e limites já existem em $R_t$?

- Quais fatos estão disponíveis em $I_t$ no instante exato da decisão?

- Quais previsões, *scores* ou probabilidades formam $B_t$?

- O que será revelado apenas depois da ação, como parte de $W_{t+1}$?
