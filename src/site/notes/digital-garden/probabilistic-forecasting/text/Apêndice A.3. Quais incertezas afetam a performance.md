---
{"dg-publish":true,"permalink":"/digital-garden/probabilistic-forecasting/text/apendice-a-3-quais-incertezas-afetam-a-performance/","dg-note-properties":{}}
---


# Apêndice A.3 — Quais incertezas afetam a performance?

Antes de escrever qualquer linha de código ou escolher uma arquitetura — XGBoost, redes neurais, regressão logística, *conformal prediction* ou um *solver* de otimização — estruture o problema em sua complexidade real.

A premissa central é: separe a decisão — a alavanca sob seu controle — do resultado, que acontece depois sob influência do ambiente e da aleatoriedade.

Este Framework não é novidade alguma. Ela é apenas o Framework já conhecido de Warren Powell para otimização de sistemas, apenas com uma vestimenta diferente observada para projetos de ciência de dados. Ao contrário de outros frameworks, acredito que o dele resolve uma dor que sinto em alguns frameworks projetados em Machine Learning: focam na solução, não no negócio. Acredito que se tivermos um design, um passo antes de tudo, pensando no problema como negócio e o que rege ele, o modelo passa a traduzir melhor para diretores e C-level.

Outro ponto importante a comentar, é que estruturei este texto propositalmente para ser utilizado junto a qualquer LLM, para ele ser um assistente à discussão.

<br>

## Passo 3: Quais incertezas afetam a performance?

Na ciência de dados tradicional, equipes costumam tratar a incerteza apenas como o erro residual do modelo ($\epsilon$) ou o desvio padrão de um treino. No mundo real da tomada de decisão, a incerteza é dinâmica, temporal e frequentemente dependente da própria decisão que tomamos.

Não basta saber "o que não sabemos". É obrigatório mapear quando, como e sob qual atraso a incerteza se resolve na linha do tempo operacional.

<br>

## A Regra da Revelação Temporal em Ciência de Dados

Toda incerteza opera sob uma fronteira temporal estrita:

1. **Antes da Ação ($t$):** A incerteza reside no modelo como uma crença estatística imperfeita ($B_t$).

2. **No Momento da Ação ($t$):** Você puxa a alavanca e toma a decisão $a_t$ baseado no estado $S_t = (R_t, I_t, B_t)$.

3. **Após a Ação ($t+1$ a $t+H$):** O mundo reage e revela o choque exógeno ou ruído estocástico ($W_{t+1}$).

> **O Teste da Revelação Temporal:** _Essa informação está disponível para a política antes da ação ou só é observada depois que o sistema a executa?_ Se estiver disponível antes, é informação observada em $I_t$, ainda que sujeita a problemas de qualidade. Se só se revelar depois, é parte do risco $W_{t+1}$ que a política precisa suportar.

<br>

### Tipos Fundamentais de Incerteza

| **Tipo de Incerteza**                                       | **O que representa na Operação**                                                                   | **Exemplos Reais em Projetos de ML**                                                                                                                                                                    |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Incerteza de Resposta Comportamental                        | A reação estocástica do ser humano à decisão que o modelo tomou.                                   | O cliente converte ou ignora o desconto ofertado? O usuário clica no item recomendado na posição 1? O lojista aceita a taxa recalculada pela precificação dinâmica?                                     |
| Incerteza de Rótulo Atrasado                                | O tempo decorrido entre a decisão e a confirmação do rótulo real (_target_).                       | Em fraude de cartão, a contestação (_chargeback_) só chega entre 30 e 90 dias após a aprovação ($t+90$). Em crédito, a inadimplência só se confirma após 60 dias de atraso na fatura.                   |
| Incerteza de Censura Contrafactual                          | O resultado real é ocultado porque a própria decisão bloqueou a observação.                        | Se o modelo reprovar um proponente de empréstimo ou bloquear preventivamente uma conta, o sistema nunca saberá se ele realmente cometeria fraude ou pagaria a dívida.                                   |
| Incerteza de Degradação do Ambiente (_Drift_)               | Mudança exógena na distribuição dos dados ou na relação causal entre variáveis.                    | Mudanças na taxa básica de juros que alteram subitamente o comportamento de inadimplência (_Concept Drift_); lançamento de um novo iPhone que altera os padrões de compra do e-commerce (_Data Drift_). |
| Incerteza Epistêmica e Amostras OOD (_Out-of-Distribution_) | A falta de conhecimento do próprio modelo ao receber dados de regiões onde ele nunca foi treinado. | Um modelo de concessão de crédito para MEI recebendo uma empresa de faturamento 100 vezes maior que a base de treino; o modelo emite um score alto sem base estatística real (excesso de confiança).    |
| Incerteza de Pipeline e Latência de Engenharia              | Atrasos na ingestão e assincronia que geram discrepância entre o fato no mundo e o dado no modelo. | _Feature Store_ desatualizada consumindo dados de saldo de 2 horas atrás em vez do saldo em tempo real; falha silenciosa de API externa de bureaus de crédito retornando nulo ou valor padrão.          |

<br>

## Roteiro para discussão

Use estas perguntas para enquadrar o problema com o time ou orientar uma LLM:

- O que é conhecido antes da decisão e o que só será revelado depois dela?

- Em que instante cada incerteza se resolve e com qual atraso o feedback chega?

- A ação escolhida altera o que poderá ser observado depois, criando censura ou viés de seleção?

- Qual incerteza ameaça mais a decisão: comportamento, *drift*, dados fora de distribuição ou falha de pipeline?

- Como a política se protege: intervalo, limite, regra de contingência, experimento ou revisão humana?
