# agent.py
# Este arquivo define o AGENTE DQN:
# - política epsilon-greedy
# - replay buffer
# - atualização Q-learning
# - target network
# - contadores e checkpoint

from __future__ import annotations  # permite usar tipos da própria classe

import numpy as np              # aleatoriedade e arrays
import torch                    # PyTorch
import torch.nn as nn           # loss functions

from config import CFG          # configuração global
from data import ReplayBuffer   # replay buffer
from dqn import DQN             # rede neural DQN


# =========================
# Epsilon Schedule
# =========================
class EpsilonSchedule:
    """
    Controla o decaimento do epsilon (exploração).

    IMPORTANTE:
    - epsilon é medido em FRAMES do ambiente
    - não em decisões
    """

    def __init__(self, start: float, end: float, decay_frames: int):
        # Valor inicial do epsilon (ex: 1.0)
        self.start = float(start)

        # Valor final mínimo do epsilon (ex: 0.1)
        self.end = float(end)

        # Quantidade total de frames para decair
        self.decay_frames = int(decay_frames)

        # Contador de frames já vistos
        self.frames = 0  # ENV FRAMES

    def value(self) -> float:
        """
        Retorna o valor atual do epsilon.
        """
        # Se decay inválido, fixa no mínimo
        if self.decay_frames <= 0:
            return self.end

        # Fração do decaimento já percorrida
        t = min(1.0, self.frames / float(self.decay_frames))

        # Interpolação linear entre start e end
        return self.start + t * (self.end - self.start)


# =========================
# Agente DQN
# =========================
class DQNAgent:
    """
    Implementa o agente DQN clássico (paper 2013).
    """

    def __init__(self, n_actions: int, device: str):
        # Dispositivo de execução (cpu ou cuda)
        self.device = torch.device(device)

        # Número de ações possíveis no ambiente
        self.n_actions = int(n_actions)

        # -------------------------
        # Redes neurais
        # -------------------------

        # Rede principal Q(s,a)
        self.q_net = DQN(self.n_actions).to(self.device)

        # Rede alvo (target network)
        self.target_net = DQN(self.n_actions).to(self.device)

        # Inicialmente, target = q_net
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Target network nunca treina por gradiente
        self.target_net.eval()

        # -------------------------
        # Otimizador (paper)
        # -------------------------
        self.optimizer = torch.optim.RMSprop(
            self.q_net.parameters(),  # parâmetros da Q-network
            lr=CFG.LR,                # learning rate
            alpha=CFG.RMSPROP_ALPHA,  # média móvel
            eps=CFG.RMSPROP_EPS,      # estabilidade numérica
        )

        # -------------------------
        # Replay Buffer
        # -------------------------
        self.replay = ReplayBuffer(
            CFG.REPLAY_CAPACITY,
            CFG.STACK_K,
            CFG.IMG_SIZE
        )

        # Gerador de números aleatórios (reprodutível)
        self.rng = np.random.default_rng(0)

        # -------------------------
        # Epsilon-greedy
        # -------------------------
        self.eps = EpsilonSchedule(
            CFG.EPS_START,
            CFG.EPS_END,
            CFG.EPS_DECAY_FRAMES
        )

        # Contador de decisões tomadas
        self.decisions = 0

        # Função de perda (paper usa MSE)
        self.loss_fn = nn.MSELoss()

        # Flag para imprimir apenas a primeira loss
        self._printed_first_loss = False

        # -------------------------
        # Sanidade da rede (1x)
        # -------------------------
        if CFG.DEBUG_ONCE:
            with torch.no_grad():
                # Input fake só para validar shapes
                x = torch.zeros((1, 4, 84, 84), dtype=torch.uint8, device=self.device)
                y = self.q_net(x)
                print("dqn_sanity:", tuple(y.shape))

    # =========================
    # Seleção de ação (treino)
    # =========================
    @torch.no_grad()
    def select_action_train(self, state: np.ndarray) -> int:
        """
        Seleciona ação usando epsilon-greedy (modo treino).
        """
        # Exploração: ação aleatória
        if self.rng.random() < self.eps.value():
            return int(self.rng.integers(0, self.n_actions))

        # Exploração desligada: usa a Q-network
        s = torch.from_numpy(state).unsqueeze(0).to(self.device)
        q = self.q_net(s)

        # Escolhe ação com maior Q-value
        return int(torch.argmax(q, dim=1).item())

    # =========================
    # Seleção de ação (avaliação)
    # =========================
    @torch.no_grad()
    def select_action_eval(self, state: np.ndarray) -> int:
        """
        Seleciona ação no modo avaliação (play).
        """
        # Pequena exploração residual
        if self.rng.random() < float(CFG.EPS_EVAL):
            return int(self.rng.integers(0, self.n_actions))

        s = torch.from_numpy(state).unsqueeze(0).to(self.device)
        q = self.q_net(s)
        return int(torch.argmax(q, dim=1).item())

    # =========================
    # Armazenar transição
    # =========================
    def store(self, s, a, r, ns, d):
        """
        Armazena uma transição no replay buffer.
        """
        self.replay.push(s, a, r, ns, d)

    # =========================
    # Passo de treinamento
    # =========================
    def train_step(self):
        """
        Executa UM passo de Q-learning.
        Retorna a loss ou None.
        """
        # Não treina antes do warmup
        if len(self.replay) < int(CFG.REPLAY_WARMUP):
            return None

        # Treina apenas a cada N decisões
        if (self.decisions % int(CFG.TRAIN_EVERY_DECISION)) != 0:
            return None

        # Amostra batch do replay
        s, a, r, ns, d = self.replay.sample(int(CFG.BATCH_SIZE), self.rng)

        # Converte para tensores
        s_t = torch.from_numpy(s).to(self.device)
        ns_t = torch.from_numpy(ns).to(self.device)
        a_t = torch.from_numpy(a).long().to(self.device)
        r_t = torch.from_numpy(r).float().to(self.device)
        d_t = torch.from_numpy(d).float().to(self.device)

        # Q(s,a) atual
        q_sa = self.q_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

        # Target Q-learning (sem gradiente)
        with torch.no_grad():
            q_next = self.target_net(ns_t).max(1).values
            target = r_t + (1.0 - d_t) * float(CFG.GAMMA) * q_next

        # Calcula a loss (MSE)
        loss = self.loss_fn(q_sa, target)

        # Sanidade numérica
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss não-finita: {loss.item()}")

        # Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # Log da primeira loss
        if (not self._printed_first_loss) and CFG.DEBUG_ONCE:
            print("first_loss:", float(loss.item()))
            self._printed_first_loss = True

        return float(loss.item())

    # =========================
    # Contadores e target net
    # =========================
    def step_counters(self):
        """
        Atualiza contadores globais do agente.
        """
        # Incrementa decisões
        self.decisions += 1

        # Incrementa frames (epsilon é em frames)
        self.eps.frames += int(CFG.FRAME_SKIP)

        # Logs de sanidade pontuais
        if self.decisions in (1, 2000, 12500) and CFG.DEBUG_ONCE:
            print(
                "counters_ok:",
                "dec", self.decisions,
                "eps_frames", self.eps.frames,
                "eps", round(self.eps.value(), 4)
            )

        # Atualização da target network (hard copy)
        if (self.decisions % int(CFG.TARGET_UPDATE_EVERY_DECISION)) == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # =========================
    # Checkpoint
    # =========================
    def state_dict(self):
        """
        Retorna tudo que é necessário para retomar o treino.
        """
        return {
            "q": self.q_net.state_dict(),          # pesos da Q-network
            "t": self.target_net.state_dict(),     # pesos da target network
            "opt": self.optimizer.state_dict(),    # estado do otimizador
            "dec": int(self.decisions),            # decisões já tomadas
            "eps": int(self.eps.frames),           # frames já vistos
            "rng": self.rng.bit_generator.state,   # estado do RNG
        }

    def load_state_dict(self, ckpt):
        """
        Restaura o estado completo do agente.
        """
        self.q_net.load_state_dict(ckpt["q"])
        self.target_net.load_state_dict(ckpt["t"])
        self.optimizer.load_state_dict(ckpt["opt"])

        self.decisions = int(ckpt.get("dec", 0))
        self.eps.frames = int(ckpt.get("eps", 0))

        # Restaura o estado do gerador aleatório
        st = ckpt.get("rng", None)
        if st is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = st
