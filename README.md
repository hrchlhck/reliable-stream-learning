# Reliable Stream Learning 
A Reliable Stream Learning Intrusion Detection Model for Delayed Updates

- Compartamento do tráfego da rede muda com o tempo
- Dificuldade para atualização
- Label normalmente se torna disponível após uma janela de tempo, sistema permanece desatualizado e exposto
	
### Proposta
- Pool de classificadores de streaming (adaptive, ozabagging, boosting, hoeffding, ...)
- Rejeição por classificador
- Voto entre aceitos
- Atualização com rejeitados com 1 mês de atraso (em fevereiro, atualiza com eventos rejeitados em janeiro)

### Testes
- [X] Avaliação das técnicas tradicionais batch (RF, Boosting, MLP)
- [X] Avaliação das técnicas tradicionais batch com atualização mensal com 1 mês de atraso (RF, Boosting, MLP)
  - [X] Avaliar custo computacional e acurácia das duas acima
- [X] Avaliação das técnicas individuais streaming, sem atualização
  - [X] Avaliar custo computacional e acurácia
- [X] Avaliação da abordagem proposta com votação entre os streamings
  - [X] Avaliar custo computacional e acurácia
- [X] Avaliação da abordagem proposta com rejeição entre os streaming sem atualização
  - Avaliar custo computacional e acurácia
- [X] Avaliação da abordagem proposta com rejeição entre os streaming com atualização após 30 dias
  - [X] Avaliar custo computacional e acurácia

### Base utilizada
- [MAWILab Dataset](https://secplab.ppgia.pucpr.br/?q=idsovertime)
