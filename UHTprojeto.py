import cv2
from ultralytics import solutions, YOLO
import logging
import torch
import queue
import re
from paddleocr import PaddleOCR
import threading
import os
from datetime import datetime

push_log_http("🧠 OCR online e enviando logs para o painel")

# ========== CONFIGURAÇÃO DE LOGGING ==========
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)


import requests
FLASK_URL = "http://127.0.0.1:5000"

def push_log_http(msg: str):
    try:
        r = requests.post(f"{FLASK_URL}/push_log", json={"msg": str(msg)}, timeout=1.5)
        # opcional: print se der erro HTTP
        if r.status_code >= 400:
            print("push_log_http HTTP", r.status_code, r.text)
    except Exception as e:
        print("push_log_http falhou:", e)

try:
    paddleocr = PaddleOCR(
        lang="en",
        show_log=False,
        use_gpu=torch.cuda.is_available(),
        enable_mkldnn=False,
        log_level="ERROR"
    )
    print("✅ PaddleOCR inicializado")
except Exception as e:
    print(f"❌ Erro no PaddleOCR: {e}")
    paddleocr = None

# Variáveis globais
ocr_queue = queue.Queue()
target_dates = ['251124', '210925']  # Datas de produção e validade
target_code = '2024330'  # Código do lote

# ========== VARIÁVEIS DE CONTROLE DE ERROS ==========
# Contadores consecutivos (são zerados quando acertam)
date1_consecutive_errors = 0
date2_consecutive_errors = 0
lote_consecutive_errors = 0

# Contadores permanentes (NUNCA são zerados)
date1_total_errors = 0
date2_total_errors = 0
lote_total_errors = 0

successful_images = 0
shutdown_flag = False

# Diretórios
SUCCESS_FRAMES_DIR = "success_frames"
os.makedirs(SUCCESS_FRAMES_DIR, exist_ok=True)


def ocr_worker():
    global date1_consecutive_errors, date2_consecutive_errors, lote_consecutive_errors
    global date1_total_errors, date2_total_errors, lote_total_errors
    global successful_images, shutdown_flag

    while not shutdown_flag:
        try:
            item = ocr_queue.get(timeout=1)
            if item is None:
                break

            image, image_filename, full_frame = item

            # Criar diretório se não existir
            os.makedirs(os.path.dirname(image_filename) if os.path.dirname(image_filename) else '.', exist_ok=True)

            cv2.imwrite(image_filename, image)

            # Extrair texto e confiança
            text, confidence = extract_text_from_image(image)

            image_fail = 0

            # Verificar se o OCR encontrou algum texto
            if text and text != '0000' and text.strip():
                numeric_sequence = re.sub(r'\D', '', text)

                date1_match = target_dates[0] in numeric_sequence
                date2_match = target_dates[1] in numeric_sequence
                lote_match = target_code in numeric_sequence

                # ========== LÓGICA DE ERROS CONSECUTIVOS ==========
                if not date1_match:
                    date1_consecutive_errors += 1
                    date1_total_errors += 1  # Contador PERMANENTE
                    image_fail += 1
                    if date1_consecutive_errors == 2:
                        print(
                            f'❌ Erro crítico: Data de produção "{target_dates[0]}" não encontrada na imagem {image_filename} por 2 vezes consecutivas.')
                else:
                    date1_consecutive_errors = 0  # Resetar apenas o consecutivo

                if not date2_match:
                    date2_consecutive_errors += 1
                    date2_total_errors += 1  # Contador PERMANENTE
                    image_fail += 1
                    if date2_consecutive_errors == 2:
                        print(
                            f'❌ Erro crítico: Data de validade "{target_dates[1]}" não encontrada na imagem {image_filename} por 2 vezes consecutivas.')
                else:
                    date2_consecutive_errors = 0  # Resetar apenas o consecutivo

                if not lote_match:
                    lote_consecutive_errors += 1
                    lote_total_errors += 1  # contador PERMANENTE
                    image_fail += 1
                    if lote_consecutive_errors == 2:
                        print(
                            f'❌ Erro crítico: Lote "{target_code}" não encontrado na imagem {image_filename} por 2 vezes consecutivas.')
                else:
                    lote_consecutive_errors = 0  # Resetar apenas o consecutivo

                # Verificar se a imagem foi bem-sucedida
                if image_fail < 3:  # Sucesso se pelo menos uma condição estiver correta
                    successful_images += 1

                    # Salvar frame completo de sucesso
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    success_frame_filename = f"{SUCCESS_FRAMES_DIR}/success_frame_{timestamp}.jpg"
                    cv2.imwrite(success_frame_filename, full_frame)

                    push_log_http(f'✅ SUCESSO - Imagem #{successful_images}')
                    push_log_http(f'   Data 1: {target_dates[0]} - {"ok" if date1_match else "X"}')
                    push_log_http(f'   Data 2: {target_dates[1]} - {"ok" if date2_match else "X"}')
                    push_log_http(f'   Lote: {target_code} - {"ok" if lote_match else "X"}')
                    push_log_http(f'   Frame completo salvo: {success_frame_filename}')
                    push_log_http(
                        f'   Erros consecutivos: Data1={date1_consecutive_errors}, Data2={date2_consecutive_errors}, Lote={lote_consecutive_errors}'
                    )
                    push_log_http(
                        f'   Erros totais: Data1={date1_total_errors}, Data2={date2_total_errors}, Lote={lote_total_errors}'
                    )
                else:
                    push_log_http(f'❌ FALHA - Processando imagem: {image_filename}')
                    push_log_http(f'   Texto reconhecido: {text}')
                    push_log_http(f'   Sequência numérica extraída: {numeric_sequence}')
                    push_log_http(f'   Data 1: {target_dates[0]} - {"ok" if date1_match else "X"}')
                    push_log_http(f'   Data 2: {target_dates[1]} - {"ok" if date2_match else "X"}')
                    push_log_http(f'   Lote: {target_code} - {"ok" if lote_match else "X"}')
                    push_log_http('   Resultado: Falha em todas as condições.')
            else:
                # OCR não encontrou texto válido
                push_log_http(f'❌ FALHA: Nenhum texto detectado - {image_filename}')
                # Incrementar contadores consecutivos
                date1_consecutive_errors += 1
                date2_consecutive_errors += 1
                lote_consecutive_errors += 1
                # Incrementar contadores permanentes
                date1_total_errors += 1
                date2_total_errors += 1
                lote_total_errors += 1

            ocr_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Erro no worker OCR: {e}")
            ocr_queue.task_done()


def extract_text_from_image(image):
    """Extrai texto da imagem - retorna (texto, confiança_media)"""
    try:
        if paddleocr is None:
            return None, 0

        temp_path = "temp_ocr.jpg"
        cv2.imwrite(temp_path, image)
        result = paddleocr.ocr(temp_path)

        if not result or not result[0]:
            return None, 0

        full_text = ""
        confidences = []

        for line in result[0]:
            texto = line[1][0]
            confidence = line[1][1]
            full_text += texto + " "
            confidences.append(confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        return full_text.strip(), avg_confidence

    except Exception as e:
        print(f"❌ Erro no OCR: {e}")
        return None, 0
    finally:
        if os.path.exists("temp_ocr.jpg"):
            os.remove("temp_ocr.jpg")

        # ========== CONFIGURAÇÃO YOLO ==========


cap = cv2.VideoCapture("output.mp4")
assert cap.isOpened(), "Erro ao abrir vídeo"

model_path = "/home/jhmendonca/best.pt"
if not os.path.exists(model_path):
    model_path = "yolov8n.pt"

try:
    yolo_model = YOLO(model_path)
    if torch.cuda.is_available():
        yolo_model.to("cuda:0")
    print("✅ YOLO carregado")
except Exception as e:
    print(f"❌ Erro YOLO: {e}")
    exit(1)

# Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=[(250, 0), (250, 640), (640, 640), (640, 0)],
    model=yolo_model,
    conf=0.25,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

track_ids = set()
print("🚀 Iniciando detecção...")

worker_thread = threading.Thread(target=ocr_worker, daemon=True)
worker_thread.start()

try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("🎬 Fim do vídeo")
            break

        frame_resized = cv2.resize(im0, (640, 640))
        frame_processed = counter(frame_resized)

        # Verificar detecções
        try:
            if (hasattr(counter, 'track_data') and
                    counter.track_data is not None and
                    hasattr(counter.track_data, 'id') and
                    counter.track_data.id is not None):

                for obj_id in counter.track_data.id:
                    obj_id = int(obj_id.item())
                    if obj_id in track_ids:
                        continue

                    idx = torch.where(counter.track_data.data[:, 4] == obj_id)[0]
                    if len(idx) == 0:
                        continue

                    idx = idx[0].item()
                    x1, y1, x2, y2 = counter.track_data.data[idx, :4].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    object_center_x = (x1 + x2) // 2

                    if object_center_x > 200:
                        track_ids.add(obj_id)
                        frame_resized2 = cv2.resize(im0, (640, 640))
                        roi = frame_resized2[y1:y2, x1:x2]

                        if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                            os.makedirs("captured_images", exist_ok=True)
                            image_filename = f"captured_images/object_{obj_id}.jpg"
                            ocr_queue.put((roi, image_filename, im0))
                            print(f"📸 Objeto {obj_id} detectado - Enviando para OCR")
        except Exception as e:
            print(f"⚠️  Erro ao processar detecções: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrompido pelo usuário")

except Exception as e:
    print(f"❌ Erro durante processamento: {e}")

finally:
    print("🧹 Finalizando...")
    shutdown_flag = True

    # Aguardar a fila ser processada
    try:
        ocr_queue.join()
    except:
        pass

        # Sinalizar para a thread parar
    try:
        ocr_queue.put(None)
    except:
        pass

        # Aguardar thread terminar
    try:
        worker_thread.join(timeout=5)
    except:
        pass

    cap.release()
    cv2.destroyAllWindows()

    push_log_http("📊 ESTATÍSTICAS FINAIS:")
    push_log_http(f"   ✅ Imagens com sucesso: {successful_images}")
    push_log_http(f"   ❌ TOTAL Erros Data 1 ({target_dates[0]}): {date1_total_errors}")
    push_log_http(f"   ❌ TOTAL Erros Data 2 ({target_dates[1]}): {date2_total_errors}")
    push_log_http(f"   ❌ TOTAL Erros Lote ({target_code}): {lote_total_errors}")
    push_log_http(f"   Objetos detectados: {len(track_ids)}")
    push_log_http("🎉 Processamento concluído!")
