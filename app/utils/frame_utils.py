import cv2
import os


def extract_frames(video_path: str, every_n_frames: int = 20, max_frames: int = 200):
    """
    Extrai frames de um vídeo de forma segura, evitando crashes
    e tratando vídeos corrompidos ou incompatíveis.

    Args:
        video_path (str): Caminho do vídeo no disco
        every_n_frames (int): Intervalo entre frames
        max_frames (int): Limite máximo de frames extraídos

    Returns:
        list: Lista de frames como arrays (numpy arrays)

    Raises:
        Exception: Se o vídeo não puder ser processado
    """

    # 1. Verificar se o arquivo existe
    if not os.path.exists(video_path):
        raise Exception("Arquivo de vídeo não encontrado.")

    cap = cv2.VideoCapture(video_path)

    # 2. Verificar abertura do vídeo
    if not cap.isOpened():
        raise Exception("Falha ao abrir o vídeo. Arquivo corrompido ou codec não suportado.")

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 3. Validação básica do vídeo
    if frame_count <= 0:
        raise Exception("Vídeo sem frames detectáveis (pode estar corrompido).")

    if fps <= 0:
        # Fallback: vídeos sem FPS definido (comum em ferramentas antigas)
        fps = 30

    current_frame = 0
    extracted = 0

    while True:
        ret, frame = cap.read()

        if not ret:  # Fim do vídeo ou erro
            break

        # Extrair apenas frames no intervalo
        if current_frame % every_n_frames == 0:
            if frame is None or frame.size == 0:
                # Pular frame inválido
                current_frame += 1
                continue

            frames.append(frame)
            extracted += 1

        # Limite máximo de segurança
        if extracted >= max_frames:
            break

        current_frame += 1

    cap.release()

    # 4. Caso nenhum frame tenha sido obtido
    if len(frames) == 0:
        raise Exception(
            "Nenhum frame pôde ser extraído. O vídeo pode estar criptografado, corrompido ou em formato especial."
        )

    return frames
