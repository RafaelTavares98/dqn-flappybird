# data.py
# Este arquivo contém toda a lógica de:
# - preprocessamento de frames (pixels)
# - empilhamento de observações (FrameStack)
# - replay buffer (armazenamento de experiências)

from __future__ import annotations  # permite usar tipos como FrameStack dentro da própria classe

from dataclasses import dataclass  # facilita criar classes de dados
from typing import Tuple, Optional  # tipagem explícita para clareza

import numpy as np  # operações numéricas e arrays
import cv2  # OpenCV para conversão e resize de imagens


# =========================
# Reward clipping (paper)
# =========================
def clip_reward(r: float) -> int:
    """
    Aplica reward clipping conforme o paper do DQN:
    - recompensa positiva -> +1
    - recompensa negativa -> -1
    - zero -> 0

    Isso reduz variância e estabiliza o treinamento.
    """
    if r > 0:
        return 1
    if r < 0:
        return -1
    return 0


# =========================
# FrameStack
# =========================
@dataclass
class FrameStack:
    """
    Responsável por:
    - converter RGB -> grayscale
    - redimensionar para 84x84
    - aplicar 'max over last 2 frames' (paper)
    - empilhar k frames em formato (k, 84, 84)

    Tudo é mantido em uint8 para economizar memória.
    """

    # Quantidade de frames empilhados
    k: int = 4

    # Tamanho final da imagem (84x84)
    size: int = 84

    def __post_init__(self) -> None:
        """
        Inicializa o buffer interno de frames.
        É chamado automaticamente após o __init__ do dataclass.
        """
        self._stack = np.zeros(
            (self.k, self.size, self.size),  # (4, 84, 84)
            dtype=np.uint8                   # uint8 para economizar RAM
        )

    def reset(self, first_rgb: np.ndarray) -> np.ndarray:
        """
        Reinicia o stack no começo de um episódio.

        O primeiro frame processado é copiado
        para todas as posições do stack.
        """
        # Preprocessa o frame inicial
        p = self._process_rgb(first_rgb)

        # Preenche todo o stack com o mesmo frame
        self._stack[:] = p

        # Retorna uma cópia (evita efeitos colaterais)
        out = self._stack.copy()

        # Sanidade: formato e tipo corretos
        assert out.shape == (self.k, self.size, self.size)
        assert out.dtype == np.uint8

        return out

    def step(self, curr_rgb: np.ndarray, prev_rgb: Optional[np.ndarray]) -> np.ndarray:
        """
        Atualiza o stack a cada passo do ambiente.

        Implementa o 'max over last 2 frames' do paper:
        - reduz flickering
        - melhora estabilidade visual
        """
        # Preprocessa o frame atual
        curr_p = self._process_rgb(curr_rgb)

        # Se houver frame anterior, aplica max pixel-a-pixel
        if prev_rgb is not None:
            prev_p = self._process_rgb(prev_rgb)
            p = np.maximum(curr_p, prev_p)
        else:
            # Caso inicial (sem frame anterior)
            p = curr_p

        # Desloca o stack (FIFO)
        self._stack[:-1] = self._stack[1:]

        # Insere o frame mais recente na última posição
        self._stack[-1] = p

        # Retorna uma cópia segura
        out = self._stack.copy()

        # Sanidade: formato e tipo corretos
        assert out.shape == (self.k, self.size, self.size)
        assert out.dtype == np.uint8

        return out

    def _process_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Converte um frame RGB bruto em:
        - grayscale
        - resize para 84x84
        - uint8
        """
        # Validação defensiva do frame
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray) or frame_rgb.size == 0:
            raise ValueError("Frame RGB inválido ou vazio no preprocess.")

        # Converte RGB -> grayscale
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Redimensiona para 84x84 usando interpolação adequada
        resized = cv2.resize(
            gray,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA
        )

        # Converte explicitamente para uint8
        out = resized.astype(np.uint8)

        # Sanidade final
        assert out.shape == (self.size, self.size)
        assert out.dtype == np.uint8

        return out


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    """
    Replay Buffer implementado como ring buffer (circular).

    Armazena:
    - state e next_state como uint8 (economia de memória)
    - ações, recompensas e flags de término
    """

    def __init__(self, capacity: int, k: int, size: int):
        # Capacidade máxima do buffer
        self.capacity = int(capacity)

        # Dimensões do estado
        self.k = int(k)
        self.size = int(size)

        # Estados atuais (s)
        self.states = np.zeros(
            (self.capacity, self.k, self.size, self.size),
            dtype=np.uint8
        )

        # Próximos estados (s')
        self.next_states = np.zeros(
            (self.capacity, self.k, self.size, self.size),
            dtype=np.uint8
        )

        # Ações tomadas
        self.actions = np.zeros((self.capacity,), dtype=np.int64)

        # Recompensas (clipped)
        self.rewards = np.zeros((self.capacity,), dtype=np.int8)

        # Flags de término de episódio
        self.dones = np.zeros((self.capacity,), dtype=np.uint8)

        # Índice atual de inserção
        self._idx = 0

        # Flag indicando se o buffer já deu a volta completa
        self._full = False

    def __len__(self) -> int:
        """
        Retorna a quantidade de transições válidas no buffer.
        """
        return self.capacity if self._full else self._idx

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Insere uma nova transição no replay buffer.
        """
        i = self._idx

        # Armazena cada componente
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = int(action)
        self.rewards[i] = int(reward)
        self.dones[i] = 1 if done else 0

        # Avança o índice circular
        self._idx += 1

        # Se atingir a capacidade, volta para o início
        if self._idx >= self.capacity:
            self._idx = 0
            self._full = True

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Amostra aleatoriamente um batch do replay buffer.
        """
        n = len(self)
        if n < batch_size:
            raise ValueError("ReplayBuffer.sample chamado sem dados suficientes.")

        # Gera índices aleatórios válidos
        idxs = rng.integers(0, n, size=batch_size, endpoint=False)

        # Coleta os dados
        s = self.states[idxs]
        a = self.actions[idxs]
        r = self.rewards[idxs].astype(np.float32)
        ns = self.next_states[idxs]
        d = self.dones[idxs].astype(np.float32)

        # Sanidade de shapes e tipos
        B = batch_size
        assert s.shape == (B, self.k, self.size, self.size)
        assert ns.shape == (B, self.k, self.size, self.size)
        assert s.dtype == np.uint8
        assert ns.dtype == np.uint8
        assert a.shape == (B,) and a.dtype == np.int64
        assert r.shape == (B,) and r.dtype == np.float32
        assert d.shape == (B,) and d.dtype == np.float32

        return s, a, r, ns, d
