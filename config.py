# config.py
# Arquivo central de configuração do projeto DQN
# Aqui ficam TODOS os hiperparâmetros e escolhas de engenharia

from dataclasses import dataclass  # facilita criar classes só de configuração
import os  # usado para montar caminhos absolutos


@dataclass(frozen=True)
class Config:
    # =========================
    # Ambiente (Gym)
    # =========================

    # ID do ambiente Gymnasium que será usado
    ENV_ID: str = "FlappyBird-v0"

    # Se o ambiente deve usar observações alternativas (ex: lidar)
    # False = usamos pixels (visão)
    USE_LIDAR: bool = False

    # =========================
    # Observação / Preprocessamento
    # =========================

    # Quantidade de frames empilhados (paper usa 4)
    STACK_K: int = 4

    # Tamanho final da imagem (84x84 no paper)
    IMG_SIZE: int = 84

    # Action repeat / frame skip
    # Cada decisão é repetida por 4 frames (paper Atari)
    FRAME_SKIP: int = 4

    # =========================
    # Replay Buffer
    # =========================

    # Capacidade máxima do replay buffer
    # 1 milhão no paper original
    REPLAY_CAPACITY: int = 1_000_000

    # Quantidade mínima de transições antes de começar a treinar
    # Enquanto não atingir isso, loss = NA
    REPLAY_WARMUP: int = 50_000

    # Tamanho do batch amostrado do replay
    BATCH_SIZE: int = 32

    # =========================
    # Parâmetros do DQN
    # =========================

    # Fator de desconto (gamma)
    # Quanto o futuro importa
    GAMMA: float = 0.99

    # Learning rate do otimizador
    LR: float = 0.00025

    # =========================
    # RMSProp (igual ao paper)
    # =========================

    # Parâmetro alpha do RMSProp (média móvel)
    RMSPROP_ALPHA: float = 0.95

    # Epsilon numérico para estabilidade
    RMSPROP_EPS: float = 0.01

    # =========================
    # Agenda de treinamento
    # =========================

    # Treina a rede a cada N decisões
    # Paper: a cada 4
    TRAIN_EVERY_DECISION: int = 4

    # Frequência de atualização da target network (hard update)
    TARGET_UPDATE_EVERY_DECISION: int = 10_000

    # =========================
    # Política epsilon-greedy
    # =========================

    # Epsilon inicial (100% exploração)
    EPS_START: float = 1.0

    # Epsilon mínimo (exploração residual)
    EPS_END: float = 0.1

    # Quantidade de FRAMES (não decisões) para decair epsilon
    # Paper: 1 milhão de frames
    EPS_DECAY_FRAMES: int = 1_000_000

    # Epsilon usado no modo avaliação (play)
    EPS_EVAL: float = 0.05

    # =========================
    # Execução / Runtime
    # =========================

    # Número total de decisões do treino
    TOTAL_DECISIONS: int = 200_000

    # Frequência de log no terminal
    LOG_EVERY_DECISION: int = 2_000

    # Frequência de salvamento automático do checkpoint
    SAVE_EVERY_DECISION: int = 50_000

    # =========================
    # Debug
    # =========================

    # Se True, imprime mensagens de sanidade apenas uma vez
    DEBUG_ONCE: bool = True

    # =========================
    # Checkpoint
    # =========================

    # Nome do arquivo de checkpoint
    CKPT_NAME: str = "dqn_flappy.pt"

    # Caminho absoluto do checkpoint
    # Evita bugs de path relativo
    @property
    def CKPT_PATH(self) -> str:
        # Diretório onde este arquivo config.py está
        here = os.path.dirname(os.path.abspath(__file__))

        # Caminho final do checkpoint
        return os.path.join(here, self.CKPT_NAME)


# Instância global da configuração
# Importada em todo o projeto como CFG
CFG = Config()
