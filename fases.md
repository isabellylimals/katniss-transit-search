# 🪐 Planejamento — Detecção de Trânsitos Planetários

## Fase 1 — Montar o Dataset

 **Status:** Em andamento  

- [x] Carregar curva de luz (`lightkurve`)  
- [x] Remover valores **NaN** e **outliers**  
- [x] Aplicar **detrending** (TLS remove inclinação)  
- [x] Usar **TLS** para extrair período, `T0` e duração  
- [ ] Criar **global_view** (curva inteira reamostrada)  
- [ ] Criar **local_view** (zoom do trânsito alinhado)  
- [ ] Normalizar curvas  
- [ ] Salvar `global_view.npy` e `local_view.npy`  
- [ ] Gerar `metadata.csv` com metadados por KOI  
- [ ] Gerar `dataset_final.csv` cruzando labels do KOI  
- [ ] Revisar rótulos (classes desbalanceadas?)  
- [ ] Calcular `class_weights` (útil na loss da CNN)  
- [ ] Armazenar dados em branch separada (`tls-generation`)  
- [ ] Sincronizar e consolidar `dataset_final.csv`  

---

## Fase 2 — Preparar a Rede Neural (CNN)

🟡 **Status:** A iniciar  

- [ ] Criar `train.py` com carregamento dos `.npy`  
- [ ] Criar `Dataset` e `DataLoader` (PyTorch ou Keras)  
- [ ] Construir arquitetura simples de **CNN 1D (baseline)**  
- [ ] Aplicar `class_weights` na função de perda  
- [ ] Implementar **split** de treino/validação/teste (ex.: 70/15/15)  
- [ ] Incluir métricas de **acurácia** e **F1-score por classe**  

---

## Fase 3 — Visualização e Análise

🔴 **Status:** Pendente  

- [ ] Visualizar curvas por classe (`view_npy.py`)  
- [ ] Plotar distribuições de classe  
- [ ] Salvar gráficos em `/plots/`  
- [ ] Gerar **curvas ROC** ou **matriz de confusão**  

---

## Fase 4 — Testes, Expansão e Produção

🔴 **Status:** Pendente  

- [ ] Expandir para **500+ KOIs** com script automático  
- [ ] Automatizar pipeline completo (Makefile, bash, etc.)  
- [ ] Testar com **dados ruidosos** (simulados e reais)  
- [ ] Documentar o projeto (`README.md`, exemplos, instruções)  
- [ ] Publicar repositório (dados ignorados pelo `.gitignore`)  

---

## Fase Extra — Organização e Automação

🟡 **Status:** Parcial  

- [x] Adicionar **Makefile** para rodar as etapas (`make dataset`, `make train`, etc.)  
- [ ] Manter notebooks de exploração em `/notebooks/`  
- [ ] Criar `requirements.txt` e/ou `environment.yml`  
- [ ] Versionar dataset (`dataset/v1.0.0`)  
- [ ] Adicionar logs de execução e resultados  

---

 **Última atualização:** _(coloque a data aqui)_  **Responsável:** _(seu nome ou equipe)_  
 **Repositório:** _(link do GitHub, se houver)_
