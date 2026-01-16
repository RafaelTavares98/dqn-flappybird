# DQN Clássico (2013) — Flappy Bird 🐦

Implementação **fiel ao paper _Playing Atari with Deep Reinforcement Learning (2013)_**, aplicada ao **Flappy Bird** usando **pixels crus** e **Deep Q-Network (DQN)** em PyTorch.

O objetivo principal é **entender, reproduzir e validar o DQN clássico** e estudar a otimização para melhorar o desempenho no jogo.  
No momento, **o primeiro treino completo já foi executado** e o foco atual é **otimização e experimentação controlada**.

---

## DQN

- A rede recebe um **estado** `s` (stack de 4 frames) e devolve **Q-values**:
  - `Q(s, pular)` e `Q(s, não_pular)`
- A ação é escolhida por **argmax** (ou aleatória via epsilon):
  - `a = argmax_a Q(s,a)`
- Durante o treino, o alvo (target) segue Bellman:
  - `target = r + γ * max_a' Q_target(s', a')` (se não terminou o episódio)
- Componentes essenciais:
  - **Replay Buffer**: guarda `(s, a, r, s', done)`
  - **Q-net (online)**: aprende por gradiente
  - **Target net**: cópia atrasada da Q-net, usada para estabilizar o target

---

## Arquitetura da rede (paper 2013 — small)

Entrada: `(4, 84, 84)`  
Conv1: `16` filtros `8×8`, stride `4`  
Conv2: `32` filtros `4×4`, stride `2`  
FC: `256`  
Saída: `|A|` Q-values (um por ação)

---

## Estrutura do projeto

```
.
├── config.py        # hiperparâmetros e paths
├── requirements.txt # requerimentos
├── data.py          # FrameStack + ReplayBuffer
├── dqn.py           # rede DQN (arquitetura)
├── agent.py         # agente (epsilon, treino, target net)
├── run.py           # loop de treino e modo play
├── dqn_flappy.pt    # checkpoint (gerado)
└── README.md
```

---

## Instalação

```bash
pip install requirements.txt
```

---

## Como usar

### 1) Treinar

```bash
python run.py --mode train
```

Notas rápidas:
- Antes do **warmup** completar, a `loss` pode aparecer como `NA` (esperado).
- Interromper com `CTRL+C` deve salvar checkpoint e encerrar de forma limpa.

### 2) Jogar (play)

```bash
python run.py --mode play
```

Opcional:

```bash
python run.py --mode play --episodes 10 --fps 30
```

---

## Status

- ✅ Pipeline validada (preprocess → replay → treino → checkpoint → play)
- ✅ Baseline fiel ao paper
- 🧪 Agora: **otimização da rede e ajustes controlados** (fora do paper)

---

## Observação sobre performance

Resultado ruim pode acontecer mesmo com bastante treino, porque:
- Flappy Bird é sensível a timing fino
- `FRAME_SKIP` pode atrapalhar controle
- reward clipping reduz sinal

Este repositório prioriza **fidelidade + engenharia + compreensão**.

---

## Referência

Mnih et al. (2013) — *Playing Atari with Deep Reinforcement Learning*  
https://arxiv.org/abs/1312.5602
