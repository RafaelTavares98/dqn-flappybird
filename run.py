# run.py
# Este arquivo é o ORQUESTRADOR do projeto:
# - cria o ambiente
# - controla treino e play
# - gerencia render, episódios, logs e checkpoints
# - trata Ctrl+C com salvamento seguro

import argparse                  # leitura de argumentos da CLI
import os                        # paths e validações de arquivo
import time                      # controle de tempo / FPS
from typing import Optional, Tuple

import gymnasium as gym          # interface de ambientes
import flappy_bird_gymnasium     # registra o ambiente FlappyBird-v0
import numpy as np               # arrays numéricos
import cv2                       # exibição de frames no modo play
import torch                     # PyTorch

from config import CFG           # configurações globais
from data import FrameStack, clip_reward
from agent import DQNAgent


# ============================================================
# Render seguro em RGB
# ============================================================
def render_rgb(env) -> np.ndarray:
    """
    Garante que sempre retornamos um frame RGB válido.
    Usa cache para evitar crash caso env.render() falhe.
    """

    # Cache interno no próprio env (hack seguro)
    if not hasattr(env, "_last_rgb"):
        env._last_rgb = None

    # Chamada padrão do render
    frame = env.render()

    # Debug 1x: confirma tipo, shape e dtype
    if not hasattr(render_rgb, "_once") and CFG.DEBUG_ONCE:
        print(
            "render_ok:",
            type(frame),
            getattr(frame, "shape", None),
            getattr(frame, "dtype", None),
            getattr(frame, "size", None),
        )
        render_rgb._once = True

    # Caso normal: frame válido
    if isinstance(frame, np.ndarray) and frame.size > 0:
        env._last_rgb = frame
        return frame

    # Fallback: reutiliza último frame válido
    if env._last_rgb is not None:
        return env._last_rgb

    # Caso extremo: nada disponível
    raise RuntimeError("env.render() retornou frame vazio/None e não há cache.")


# ============================================================
# Reset de episódio robusto
# ============================================================
def reset_episode(env, stack: FrameStack) -> np.ndarray:
    """
    Reseta o ambiente e garante que o primeiro frame seja válido.
    Preenche o FrameStack com o frame inicial.
    """

    env.reset()

    try:
        # Caminho normal
        frame0 = render_rgb(env)
        return stack.reset(frame0)

    except Exception:
        # Fallback: força um step para gerar superfície
        obs, r, terminated, truncated, info = env.step(0)

        if terminated or truncated:
            env.reset()

        frame1 = render_rgb(env)
        return stack.reset(frame1)


# ============================================================
# Action repeat + captura de frames
# ============================================================
def step_repeat_with_frames(
    env,
    action: int,
    k: int
) -> Tuple[float, bool, np.ndarray, Optional[np.ndarray]]:
    """
    Executa a mesma ação por k frames (FRAME_SKIP).
    Retorna:
      - reward acumulado
      - done
      - último frame RGB
      - frame anterior (para max-over-2)
    """

    total_r = 0.0
    prev_rgb = None
    last_rgb = None
    terminated = truncated = False

    # Repetição de ação (paper-style)
    for _ in range(k):
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)

        # Captura frame atual
        rgb = render_rgb(env)

        # Atualiza histórico de frames
        prev_rgb = last_rgb
        last_rgb = rgb

        # Sai cedo se episódio acabou
        if terminated or truncated:
            break

    done = bool(terminated or truncated)

    # Sanidade: sempre precisamos de um frame válido
    assert last_rgb is not None and isinstance(last_rgb, np.ndarray) and last_rgb.size > 0
    if prev_rgb is not None:
        assert prev_rgb.shape == last_rgb.shape

    return total_r, done, last_rgb, prev_rgb


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(agent: DQNAgent) -> None:
    """
    Salva todo o estado do agente em disco.
    """
    ckpt = agent.state_dict()
    torch.save(ckpt, CFG.CKPT_PATH)

    # Garante que o arquivo foi realmente criado
    assert os.path.exists(CFG.CKPT_PATH)
    assert os.path.getsize(CFG.CKPT_PATH) > 0

    print(f"saved: {CFG.CKPT_PATH} @ decision {agent.decisions}")


def load_checkpoint(agent: DQNAgent) -> None:
    """
    Carrega o estado do agente a partir do checkpoint.
    """
    ckpt = torch.load(CFG.CKPT_PATH, map_location=agent.device)
    agent.load_state_dict(ckpt)


# ============================================================
# Loop de TREINO
# ============================================================
def train() -> None:
    """
    Loop principal de treinamento do DQN.
    """

    import datetime

    # Timestamp inicial do treino
    TRAIN_START_TS = time.time()
    TRAIN_START_STR = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"train_start: {TRAIN_START_STR}")

    # Seleção automática de device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Criação do ambiente
    env = gym.make(
        CFG.ENV_ID,
        render_mode="rgb_array",
        use_lidar=CFG.USE_LIDAR
    )

    # Pré-processamento de frames
    stack = FrameStack(k=CFG.STACK_K, size=CFG.IMG_SIZE)

    # Agente DQN
    agent = DQNAgent(n_actions=env.action_space.n, device=device)

    # Estado inicial
    state = reset_episode(env, stack)

    # Métricas de episódio
    episode_reward = 0.0
    episode_len = 0
    episode_idx = 0

    # Controle de log
    last_loss: Optional[float] = None
    last_log_dec = 0

    try:
        # Loop principal (em decisões)
        while agent.decisions < CFG.TOTAL_DECISIONS:

            # Escolha de ação (epsilon-greedy)
            action = agent.select_action_train(state)

            # Executa ação com frame skip
            total_r, done, last_rgb, prev_rgb = step_repeat_with_frames(
                env,
                action,
                CFG.FRAME_SKIP
            )

            # Próximo estado (stack + max-over-2)
            next_state = stack.step(last_rgb, prev_rgb)

            # Reward clipping (paper)
            cr = clip_reward(total_r)

            # Armazena transição
            agent.store(state, action, cr, next_state, done)

            # Passo de treinamento (se aplicável)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            # Atualiza contadores globais
            agent.step_counters()

            # Atualiza métricas do episódio
            episode_reward += total_r
            episode_len += 1
            state = next_state

            # Log periódico
            if (agent.decisions - last_log_dec) >= CFG.LOG_EVERY_DECISION:
                eps = agent.eps.value()
                buf = len(agent.replay)
                elapsed = int(time.time() - TRAIN_START_TS)

                print(
                    f"decision={agent.decisions} eps={eps:.3f} "
                    f"replay={buf} loss={('%.4f' % last_loss) if last_loss is not None else 'NA'} "
                    f"ep={episode_idx} ep_len={episode_len} ep_reward={episode_reward:.2f} "
                    f"elapsed={elapsed}s"
                )
                last_log_dec = agent.decisions

            # Checkpoint periódico
            if (agent.decisions % CFG.SAVE_EVERY_DECISION) == 0:
                save_checkpoint(agent)

            # Fim de episódio
            if done:
                episode_idx += 1
                state = reset_episode(env, stack)
                episode_reward = 0.0
                episode_len = 0

    except KeyboardInterrupt:
        # Ctrl+C seguro
        print("\nCTRL+C: interrompido. Salvando checkpoint...")
        save_checkpoint(agent)

    finally:
        env.close()
        print("train ended.")


# ============================================================
# Loop de PLAY (avaliação)
# ============================================================
def play(episodes: int, fps: int) -> None:
    """
    Executa o agente treinado em modo visual.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(
        CFG.ENV_ID,
        render_mode="rgb_array",
        use_lidar=CFG.USE_LIDAR
    )

    stack = FrameStack(k=CFG.STACK_K, size=CFG.IMG_SIZE)
    agent = DQNAgent(n_actions=env.action_space.n, device=device)

    # Carrega checkpoint
    load_checkpoint(agent)
    agent.q_net.eval()

    # Controle de FPS
    dt = 1.0 / max(1, fps)

    for ep in range(episodes):
        state = reset_episode(env, stack)
        done = False
        ep_reward = 0.0

        while not done:
            # Ação quase-greedy
            action = agent.select_action_eval(state)

            total_r, done, last_rgb, prev_rgb = step_repeat_with_frames(
                env,
                action,
                CFG.FRAME_SKIP
            )

            next_state = stack.step(last_rgb, prev_rgb)
            ep_reward += total_r
            state = next_state

            # Exibe frame
            bgr = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Flappy DQN (play)", bgr)

            # Tecla q encerra
            if cv2.waitKey(1) & 0xFF == ord("q"):
                done = True

            time.sleep(dt)

        print(f"episode {ep+1}/{episodes} reward={ep_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()


# ============================================================
# Entrada principal
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "play"], required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if args.mode == "train":
        train()
    else:
        play(episodes=args.episodes, fps=args.fps)


if __name__ == "__main__":
    main()
