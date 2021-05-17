# Reliable Stream Learning 
A Reliable Stream Learning Intrusion Detection Model for Delayed Updates

- Compartamento do trafego da rede muda com o tempo
- Dificuldade para atualização
- Label normalmente se torna disponível após uma janela de tempo, sistema permanece desatualizado e exposto
	
### Proposta
- Pool de classificadores de streaming (adaptive, ozabagging, boosting, hoeffding, ...)
- Rejeição por classificador
- Voto entre aceitos
- Atualização com rejeitados com 1 mes de atraso (em fevereiro, atualiza com eventos rejeitados em janeiro)

### Testes
1. Avaliação das técnicas tradicionais batch (RF, Boosting, MLP)
2. Avaliação das técnicas tradicionais batch com atualização mensal com 1 mês de atraso (RF, Boosting, MLP)
  - Avaliar custo computacional e acuracia das duas acima
3. Avaliação das técnicas individuais streaming, sem atualização
  - Avaliar custo computacional e acuracia
4. Avaliação da abordagem proposta com votação entre os streamings
  - Avaliar custo computacional e acuracia
5. Avaliação da abordagem proposta com rejeição entre os streaming sem atualização
  - Avaliar custo computacional e acuracia
6. Avaliação da abordagem proposta com rejeição entre os streaming com atualização após 30 dias
  - Avaliar custo computacional e acuracia

### Base utilizada
- [MAWI Dataset](https://secplab.ppgia.pucpr.br/ftp/mawi/)
