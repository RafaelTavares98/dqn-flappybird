# dqn.py
# Define a arquitetura da rede neural Deep Q-Network (DQN)
# Implementação fiel ao paper "Playing Atari with Deep Reinforcement Learning" (2013)

import torch                     # Biblioteca principal de deep learning
import torch.nn as nn            # Módulos de redes neurais
import torch.nn.functional as F  # Funções auxiliares (ex: ReLU)


class DQN(nn.Module):
    """
    Arquitetura do DQN (versão pequena do paper de 2013).

    Entrada:
      - Tensor de shape (B, 4, 84, 84)
        B = batch size
        4 = frames empilhados
        84x84 = resolução do frame preprocessado

    Camadas:
      - Conv1: 16 filtros, kernel 8x8, stride 4
      - Conv2: 32 filtros, kernel 4x4, stride 2
      - FC:    256 unidades
      - Out:   n_actions (Q-value por ação)
    """

    def __init__(self, n_actions: int):
        """
        Construtor da rede.

        Parâmetros:
        - n_actions: número de ações possíveis no ambiente
        """
        # Inicializa corretamente a classe base do PyTorch
        super().__init__()

        # Primeira camada convolucional
        # Entrada:  (B, 4, 84, 84)
        # Saída:    (B, 16, 20, 20)
        self.conv1 = nn.Conv2d(
            in_channels=4,       # 4 frames empilhados
            out_channels=16,     # 16 filtros
            kernel_size=8,       # kernel 8x8
            stride=4             # stride 4 (reduz resolução)
        )

        # Segunda camada convolucional
        # Entrada:  (B, 16, 20, 20)
        # Saída:    (B, 32, 9, 9)
        self.conv2 = nn.Conv2d(
            in_channels=16,      # saída da conv1
            out_channels=32,     # 32 filtros
            kernel_size=4,       # kernel 4x4
            stride=2             # stride 2
        )

        # Camada totalmente conectada (flatten -> 256)
        # 32 * 9 * 9 vem do tamanho exato após as convoluções
        self.fc = nn.Linear(
            in_features=32 * 9 * 9,
            out_features=256
        )

        # Camada de saída
        # Retorna um Q-value para cada ação possível
        self.out = nn.Linear(
            in_features=256,
            out_features=n_actions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passo forward da rede.

        Recebe:
        - x: tensor de estados (uint8 ou float)

        Retorna:
        - Q-values para cada ação
        """

        # Se o input vier como uint8 (0–255),
        # converte para float para operações matemáticas
        if x.dtype == torch.uint8:
            x = x.float()

        # Normaliza os pixels para o intervalo [0, 1]
        # Isso melhora a estabilidade do treinamento
        x = x / 255.0

        # Aplica a primeira convolução + ReLU
        x = F.relu(self.conv1(x))

        # Aplica a segunda convolução + ReLU
        x = F.relu(self.conv2(x))

        # Flatten: transforma (B, 32, 9, 9) -> (B, 32*9*9)
        x = x.view(x.size(0), -1)

        # Camada totalmente conectada + ReLU
        x = F.relu(self.fc(x))  

        # Camada final (sem ativação)
        # Retorna Q(s, a) para cada ação
        return self.out(x)
