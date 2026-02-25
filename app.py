import os
import time
import json
import queue
import re
import threading
from datetime import datetime
from zoneinfo import ZoneInfo  # ✅ timezone Brasil
import numpy as np
import cv2
import torch
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, session
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "troque-essa-chave-em-producao")

# ========= CONFIGURAÇÃO DO VÍDEO =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fonte de vídeo:
# - Defina RTSP_URL no ambiente para usar câmera RTSP
# - Deixe vazio para usar arquivo local (videocamera.mp4 / output.mp4)
RTSP_URL = (os.environ.get("RTSP_URL") or "").strip()

if RTSP_URL.strip():
    VIDEO_SOURCE = RTSP_URL.strip()
    print("📡 Usando RTSP:", VIDEO_SOURCE)
else:
    VIDEO_SOURCE = os.path.join(BASE_DIR, "videocamera.mp4")
    if not os.path.exists(VIDEO_SOURCE):
        VIDEO_SOURCE = os.path.join(BASE_DIR, "output.mp4")
    if not os.path.exists(VIDEO_SOURCE):
        raise RuntimeError(f"Arquivo de vídeo não encontrado. Procurei: {VIDEO_SOURCE}")
    print("🎞️ Usando arquivo:", VIDEO_SOURCE)

# ========= CARREGAR YOLO =========
model_path = os.path.join(BASE_DIR, "best.pt")
model = YOLO(model_path)
if torch.cuda.is_available():
    model.to("cuda:0")

print("🔹 Usando modelo:", model_path)
print("🔹 Vídeo:", VIDEO_SOURCE)

# ========= PaddleOCR =========
try:
    paddleocr = PaddleOCR(
        lang="en",
        show_log=False,
        use_gpu=torch.cuda.is_available(),
        enable_mkldnn=False,
        log_level="ERROR",
    )
    print("✅ PaddleOCR inicializado")
except Exception as e:
    paddleocr = None
    print("❌ Erro ao inicializar PaddleOCR:", e)


# ========= LÓGICA AUTOMÁTICA DE DATAÇÃO =========
def _calc_auto_fabricacao_lote():
    """
    Fabricação = hoje (DDMMAA)
    Lote = YYYY + dia do ano (DDD) => YYYYDDD
    Ex: 28/01/2026 -> fab=280126, lote=2026028
    """
    now = datetime.now(ZoneInfo("America/Sao_Paulo"))
    doy = now.timetuple().tm_yday
    data_fab = now.strftime("%d%m%y")
    lote = f"{now.strftime('%Y')}{doy:03d}"
    return data_fab, lote


# ========= ESTADO GLOBAL =========
RUNNING = False
RUN_ID = None

cap = None
cap_lock = threading.Lock()

# contagem vídeo
total_video = 0
seen_ids = set()

# stats OCR para o painel
pass_count = 0          # Total (OCR)
error_count = 0         # Erros (totais)
last_event_text = "—"
last_ocr_text = "—"

# “datas” (modal)
# validade é definida pelo usuário no modal
dates_state = {
    "data_validade": "",
}

# ========= FILAS / THREADS =========
log_queue = queue.Queue(maxsize=4000)
ocr_queue = queue.Queue(maxsize=300)
ocr_thread = None
ocr_stop = threading.Event()

# ========= CONTADORES OCR (consecutivos e totais) =========
date1_consecutive_errors = 0
date2_consecutive_errors = 0
lote_consecutive_errors = 0
image_fail_consecutive = 0

date1_total_errors = 0
date2_total_errors = 0
lote_total_errors = 0

SUCCESS_FRAMES_DIR = os.path.join(BASE_DIR, "success_frames")
os.makedirs(SUCCESS_FRAMES_DIR, exist_ok=True)

ROI_OK_DIR = os.path.join(BASE_DIR, "roi_ok")
ROI_FAIL_DIR = os.path.join(BASE_DIR, "roi_fail")
os.makedirs(ROI_OK_DIR, exist_ok=True)
os.makedirs(ROI_FAIL_DIR, exist_ok=True)


def require_login() -> bool:
    return session.get("logged_in") is True


def push_log(msg: str):
    msg = str(msg).strip()
    if not msg:
        return
    ts = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    try:
        log_queue.put_nowait(line)
    except queue.Full:
        try:
            _ = log_queue.get_nowait()
        except:
            pass
        try:
            log_queue.put_nowait(line)
        except:
            pass


def reset_everything_for_new_run():
    global total_video, seen_ids
    global pass_count, error_count, last_event_text, last_ocr_text
    global date1_consecutive_errors, date2_consecutive_errors, lote_consecutive_errors, image_fail_consecutive
    global date1_total_errors, date2_total_errors, lote_total_errors

    total_video = 0
    seen_ids.clear()

    pass_count = 0
    error_count = 0
    last_event_text = "—"
    last_ocr_text = "—"

    date1_consecutive_errors = 0
    date2_consecutive_errors = 0
    lote_consecutive_errors = 0
    image_fail_consecutive = 0

    date1_total_errors = 0
    date2_total_errors = 0
    lote_total_errors = 0

    # limpa fila OCR
    try:
        while True:
            ocr_queue.get_nowait()
            ocr_queue.task_done()
    except queue.Empty:
        pass


def open_video_from_start():
    global cap
    with cap_lock:
        try:
            if cap is not None:
                cap.release()
        except:
            pass

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Não consegui abrir o vídeo: {VIDEO_SOURCE}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except:
            pass


# ========= PREPROCESS ROI (BINARIZA DE VERDADE) =========
def _crop_inside_border(bin_img, margin=4):
    """
    Remove automaticamente a borda preta dominante do ROI binarizado.
    Espera: texto PRETO (0) e fundo BRANCO (255)
    """
    inv = 255 - bin_img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bin_img

    h, w = bin_img.shape[:2]
    c = max(cnts, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)

    # ignora se não for uma borda grande
    if cw < 0.4 * w or ch < 0.4 * h:
        return bin_img

    x1 = max(x + margin, 0)
    y1 = max(y + margin, 0)
    x2 = min(x + cw - margin, w)
    y2 = min(y + ch - margin, h)

    return bin_img[y1:y2, x1:x2]


def _upscale_preserve_details(img_bgr, scale: int):
    """
    Upscale com preservação de definição para OCR.
    Usa Lanczos + unsharp leve para reforçar bordas sem exagerar ruído.
    """
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # Unsharp mask leve para recuperar microdetalhes após o resize.
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=0.7, sigmaY=0.7)
    sharp = cv2.addWeighted(up, 1.12, blur, -0.12, 0)
    return sharp


def _clean_binary_inverted(bin_inv):
    """
    Limpa ruído em imagem binária invertida (texto branco em fundo preto):
    - remove pontos muito pequenos
    - remove componentes grandes coladas na borda (moldura do rótulo)
    """
    if bin_inv is None or bin_inv.size == 0:
        return bin_inv

    h, w = bin_inv.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    out = np.zeros_like(bin_inv)

    min_area = max(12, int(h * w * 0.00025))
    max_border_area = max(80, int(h * w * 0.10))

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue

        # Remove componentes muito alongados (linhas de moldura/cabeçalho)
        # preservando dígitos/letras que tendem a ter razão de aspecto moderada.
        aspect = (cw / float(ch)) if ch > 0 else 999.0
        if area > max(120, int(h * w * 0.0035)) and (aspect > 14.0 or aspect < 0.07):
            continue

        touches_border = (x <= 0 or y <= 0 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1))
        if touches_border and area > max_border_area:
            continue

        out[labels == i] = 255

    # Remove uma faixa superior pequena (normalmente cabeçalho não relevante e ruidoso).
    header_h = max(0, int(h * 0.11))
    if header_h > 0:
        out[:header_h, :] = 0

    # Reforça traços após limpeza.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out


def preprocess_roi_for_ocr(roi_bgr):
    """
    Preprocessamento para OCR com foco em nitidez dos dígitos.
    Retorna imagem binária invertida (texto branco em fundo preto).
    """
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        # 1) ROI já chega redimensionado no recorte (antes do preprocess).
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 16, 240, cv2.NORM_MINMAX)

        # 2) Realce local leve após normalização
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 3) Reduz ruído preservando bordas
        gray = cv2.bilateralFilter(gray, 5, 35, 35)
        gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)

        # 4) Binarização local conservadora
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            8
        )

        # Limpa pontos isolados sem destruir os dígitos.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        # garante texto PRETO
        if (th == 0).mean() > 0.6:
            th = 255 - th

        # 5) Remove borda externa dominante
        h, w = th.shape[:2]
        dynamic_margin = max(2, int(min(h, w) * 0.04))
        th = _crop_inside_border(th, margin=dynamic_margin)

        # Reforça continuidade dos traços dos caracteres.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Inversão final solicitada: texto branco em fundo preto.
        th = cv2.bitwise_not(th)

        # Reduz ruído fino no fundo preto sem apagar os dígitos.
        th = cv2.medianBlur(th, 3)
        _, th = cv2.threshold(th, 127, 255, cv2.THRESH_BINARY)
        # Evita apagar traços finos (ex.: "1", "7") nesta etapa final.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th = _clean_binary_inverted(th)

        return th

    except Exception as e:
        push_log(f"⚠️ Erro no preprocess ROI OCR: {repr(e)}")
        return None



def extract_text_from_image(image_any):
    """Retorna (texto, conf_media). Aceita imagem 1 canal ou BGR."""
    if paddleocr is None:
        return None, 0.0

    # PNG evita perdas por compressão em texto pequeno.
    temp_path = os.path.join(BASE_DIR, "temp_ocr.png")
    try:
        cv2.imwrite(temp_path, image_any)
        result = paddleocr.ocr(temp_path)
        if not result or not result[0]:
            return None, 0.0

        texts = []
        confs = []
        for line in result[0]:
            txt = line[1][0]
            conf = float(line[1][1])
            texts.append(txt)
            confs.append(conf)

        full_text = " ".join(texts).strip()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return full_text, avg_conf

    except Exception as e:
        push_log(f"⚠️ Erro no OCR: {repr(e)}")
        return None, 0.0

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass


def _save_fail_pair(ts: str, prefix: str, roi_bgr, roi_processed):
    """
    Salva ROI original (BGR) e binarizado (processado) com prefixos:
      roi_fail / roi_notext / roi_exception
    """
    try:
        raw_path = os.path.join(ROI_FAIL_DIR, f"{prefix}_{ts}_raw.jpg")
        cv2.imwrite(raw_path, roi_bgr)
    except:
        pass

    try:
        if roi_processed is not None:
            bin_path = os.path.join(ROI_FAIL_DIR, f"{prefix}_{ts}_bin.jpg")
            cv2.imwrite(bin_path, roi_processed)
    except:
        pass



def ocr_worker():
    """
    Thread OCR: consome ROIs, roda OCR, compara com datas_state
    e salva ROI OK/FAIL (inclui binário também no FAIL).
    """
    global pass_count, error_count, last_event_text, last_ocr_text
    global date1_consecutive_errors, date2_consecutive_errors, lote_consecutive_errors, image_fail_consecutive
    global date1_total_errors, date2_total_errors, lote_total_errors
    auto_fab0, auto_lote0 = _calc_auto_fabricacao_lote()
    initial_val = (dates_state.get("data_validade") or "").strip() or "NÃO DEFINIDA"
    push_log(
        f"🎯 Alvos OCR (AUTO): fab={auto_fab0} | "
        f"val={initial_val} | lote={auto_lote0}"
    )

    while not ocr_stop.is_set():
        try:
            item = ocr_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            try:
                ocr_queue.task_done()
            except:
                pass
            break

        roi_bgr, full_frame_bgr, image_filename = item

        roi_processed = None

        try:
            auto_fab, auto_lote = _calc_auto_fabricacao_lote()
            val = (dates_state.get("data_validade") or "").strip()
            target_dates = [auto_fab, val]
            target_code = auto_lote

            # Fluxo único: preprocessa ROI e roda OCR uma vez.
            roi_processed = preprocess_roi_for_ocr(roi_bgr)
            if roi_processed is None or roi_processed.size == 0:
                roi_processed = roi_bgr
            text, confidence = extract_text_from_image(roi_processed)
            last_ocr_text = text if text else "—"

            if text and text.strip() and text != "0000":
                numeric_sequence = re.sub(r"\D", "", text)

                date1_match = target_dates[0] in numeric_sequence
                date2_match = target_dates[1] in numeric_sequence
                lote_match = target_code in numeric_sequence

                image_fail = 0

                if not date1_match:
                    date1_consecutive_errors += 1
                    date1_total_errors += 1
                    image_fail += 1
                    if date1_consecutive_errors == 2:
                        push_log(f'❌ Erro crítico: Data de produção "{target_dates[0]}" não encontrada por 2 vezes consecutivas.')
                else:
                    date1_consecutive_errors = 0

                if not date2_match:
                    date2_consecutive_errors += 1
                    date2_total_errors += 1
                    image_fail += 1
                    if date2_consecutive_errors == 2:
                        push_log(f'❌ Erro crítico: Data de validade "{target_dates[1]}" não encontrada por 2 vezes consecutivas.')
                else:
                    date2_consecutive_errors = 0

                if not lote_match:
                    lote_consecutive_errors += 1
                    lote_total_errors += 1
                    image_fail += 1
                    if lote_consecutive_errors == 2:
                        push_log(f'❌ Erro crítico: Lote "{target_code}" não encontrado por 2 vezes consecutivas.')
                else:
                    lote_consecutive_errors = 0

                # ✅ mantém sua regra original: sucesso se pelo menos 1 bateu (image_fail < 3)
                if image_fail < 3:
                    image_fail_consecutive = 0
                    pass_count += 1

                    timestamp = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S_%f")
                    success_frame_filename = os.path.join(SUCCESS_FRAMES_DIR, f"success_frame_{timestamp}.jpg")
                    try:
                        cv2.imwrite(success_frame_filename, full_frame_bgr)
                    except:
                        pass

                    # ✅ ROI_OK salva a imagem enviada ao OCR (neste teste: ROI redimensionado)
                    try:
                        roi_path = os.path.join(ROI_OK_DIR, f"roi_ok_{timestamp}_conf{confidence:.2f}.jpg")
                        cv2.imwrite(roi_path, roi_processed)
                    except:
                        pass

                    push_log(f"✅ SUCESSO - Imagem #{pass_count}")
                    push_log(f'   Data 1: {target_dates[0]} - {"ok" if date1_match else "X"}')
                    push_log(f'   Data 2: {target_dates[1]} - {"ok" if date2_match else "X"}')
                    push_log(f'   Lote: {target_code} - {"ok" if lote_match else "X"}')
                    push_log(f"   Texto OCR: {text} (conf={confidence:.2f})")
                    push_log(f"   Sequência numérica: {numeric_sequence}")
                    push_log(f"   Frame completo salvo: {success_frame_filename}")

                    last_event_text = f"✅ SUCESSO - Imagem #{pass_count}"

                else:
                    image_fail_consecutive += 1
                    push_log(f"⚠️ Caixa com falha total ({image_fail_consecutive}/3): {image_filename}")

                    if image_fail_consecutive >= 3:
                        error_count += 1
                        push_log(f"❌ FALHA - 3 caixas consecutivas com erro: {image_filename}")

                        ts = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S_%f")
                        _save_fail_pair(ts, "roi_fail", roi_bgr, roi_processed)

                        push_log(f"   Texto reconhecido: {text}")
                        push_log(f"   Sequência numérica extraída: {numeric_sequence}")
                        push_log(f'   Data 1: {target_dates[0]} - {"ok" if date1_match else "X"}')
                        push_log(f'   Data 2: {target_dates[1]} - {"ok" if date2_match else "X"}')
                        push_log(f'   Lote: {target_code} - {"ok" if lote_match else "X"}')
                        last_event_text = "❌ FALHA OCR (3 consecutivas)"
                        image_fail_consecutive = 0
                    else:
                        last_event_text = f"⚠️ Aguardando 3 falhas consecutivas ({image_fail_consecutive}/3)"
            else:
                # OCR não achou texto
                image_fail_consecutive += 1
                push_log(f"⚠️ Caixa sem texto ({image_fail_consecutive}/3): {image_filename}")

                date1_consecutive_errors += 1
                date2_consecutive_errors += 1
                lote_consecutive_errors += 1
                date1_total_errors += 1
                date2_total_errors += 1
                lote_total_errors += 1
                if image_fail_consecutive >= 3:
                    error_count += 1
                    push_log(f"❌ FALHA: 3 caixas consecutivas sem texto - {image_filename}")

                    ts = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S_%f")
                    _save_fail_pair(ts, "roi_notext", roi_bgr, roi_processed)
                    last_event_text = "❌ FALHA: Nenhum texto (3 consecutivas)"
                    image_fail_consecutive = 0
                else:
                    last_event_text = f"⚠️ Aguardando 3 falhas consecutivas ({image_fail_consecutive}/3)"

        except Exception as e:
            image_fail_consecutive += 1
            push_log(f"⚠️ Erro no worker OCR ({image_fail_consecutive}/3): {e}")

            if image_fail_consecutive >= 3:
                error_count += 1
                push_log("❌ FALHA: 3 caixas consecutivas com exceção no OCR")
                try:
                    ts = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S_%f")
                    _save_fail_pair(ts, "roi_exception", roi_bgr, roi_processed)
                except:
                    pass
                last_event_text = "❌ FALHA: Exceção OCR (3 consecutivas)"
                image_fail_consecutive = 0
            else:
                last_event_text = f"⚠️ Aguardando 3 falhas consecutivas ({image_fail_consecutive}/3)"

        finally:
            try:
                ocr_queue.task_done()
            except:
                pass

    push_log("🧹 OCR worker finalizado")


# ========= LOGIN =========
@app.route("/", methods=["GET"])
def root():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()

    USER = os.environ.get("UHT_USER", "admin")
    PASS = os.environ.get("UHT_PASS", "admin")

    if username == USER and password == PASS:
        session["logged_in"] = True
        push_log("✅ Login realizado")
        return redirect(url_for("dashboard"))

    return render_template("login.html", error="Usuário ou senha inválidos")


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "running": RUNNING})


# ========= START/STOP =========
@app.route("/start_run", methods=["POST"])
def start_run():
    global RUNNING, RUN_ID, ocr_thread

    if not require_login():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    val = (dates_state.get("data_validade") or "").strip()
    if not val:
        return jsonify({
            "ok": False,
            "error": "Defina a data de validade no modal antes de iniciar."
        }), 400

    reset_everything_for_new_run()
    open_video_from_start()

    RUNNING = True
    RUN_ID = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d%H%M%S")
    push_log(f"🚀 Sistema iniciado (run_id={RUN_ID}) | vídeo reiniciado | contadores zerados")

    # inicia OCR thread
    ocr_stop.clear()
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    return jsonify({"ok": True, "run_id": RUN_ID})


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    global RUNNING, RUN_ID

    if not require_login():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    RUNNING = False
    RUN_ID = None

    # para OCR
    ocr_stop.set()
    try:
        ocr_queue.put_nowait(None)
    except:
        pass

    push_log("🛑 Sistema parado")

    push_log("📊 ESTATÍSTICAS FINAIS:")
    push_log(f"   ✅ Imagens com sucesso: {pass_count}")
    push_log(f"   ❌ TOTAL Erros Data 1: {date1_total_errors}")
    push_log(f"   ❌ TOTAL Erros Data 2: {date2_total_errors}")
    push_log(f"   ❌ TOTAL Erros Lote: {lote_total_errors}")
    push_log(f"   Objetos detectados (vídeo): {len(seen_ids)}")
    push_log("🎉 Processamento concluído!")

    return jsonify({"ok": True})


# ========= SSE LOGS =========
@app.route("/logs")
def logs():
    def event_stream():
        yield ": ok\n\n"
        while True:
            try:
                msg = log_queue.get(timeout=10)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": ping\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


# ========= SSE STATS =========
@app.route("/stats_stream")
def stats_stream():
    def event_stream():
        yield ": ok\n\n"
        while True:
            payload = {
                "run_id": RUN_ID if RUNNING else None,
                "pass_count": pass_count,
                "error_count": error_count,
                "last_event": last_event_text,
                "last_ocr": last_ocr_text,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream")


# ========= DATAÇÃO =========
@app.route("/get_dates")
def get_dates():
    if not require_login():
        return jsonify({"error": "unauthorized"}), 401

    auto_fab, auto_lote = _calc_auto_fabricacao_lote()
    return jsonify({
        "data_fabricacao": auto_fab,
        "data_validade": (dates_state.get("data_validade") or "").strip(),
        "lote": auto_lote,
    })


@app.route("/set_dates", methods=["POST"])
def set_dates():
    if not require_login():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}

    # Apenas a validade é persistida; fabricação/lote são sempre automáticos.
    if "data_validade" in data:
        dates_state["data_validade"] = (data.get("data_validade") or "").strip()

    auto_fab, auto_lote = _calc_auto_fabricacao_lote()

    push_log(
        f"📅 Datação atualizada | fab={auto_fab} | "
        f"val={(dates_state.get('data_validade') or '').strip() or 'NÃO DEFINIDA'} | "
        f"lote={auto_lote}"
    )
    return jsonify({"ok": True})


# ========= VIDEO FEED =========
def gen_frames():
    global total_video, seen_ids

    while True:
        if cap is None:
            time.sleep(0.05)
            continue

        with cap_lock:
            ok, frame = cap.read()

        if not ok:
            if RUNNING:
                with cap_lock:
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except:
                        pass
            time.sleep(0.02)
            continue

        # Mantém o frame original para recorte de ROI em alta definição.
        frame_original = frame
        h0, w0 = frame_original.shape[:2]

        # Frame reduzido só para tracking/render.
        frame_proc = cv2.resize(frame_original, (640, 640))
        h1, w1 = frame_proc.shape[:2]
        sx = w0 / float(w1)
        sy = h0 / float(h1)
        line_x = w1 // 2

        if not RUNNING:
            annotated = frame_proc
        else:
            try:
                results = model.track(frame_proc, persist=True, verbose=False)
                r = results[0]
                annotated = r.plot()

                # linha de contagem
                cv2.line(annotated, (line_x, 0), (line_x, annotated.shape[0]), (0, 255, 255), 2)

                if r.boxes is not None and r.boxes.id is not None:
                    for box, obj_id in zip(r.boxes.xyxy, r.boxes.id):
                        obj_id = int(obj_id.item())
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cx = int((x1 + x2) / 2)

                        if obj_id not in seen_ids and cx > line_x:
                            seen_ids.add(obj_id)
                            total_video += 1
                            push_log(f"📦 Contado id={obj_id} | total_video={total_video}")

                            # ROI para OCR extraído do frame original (sem perda de definição).
                            x1o = max(0, min(w0 - 1, int(round(x1 * sx))))
                            y1o = max(0, min(h0 - 1, int(round(y1 * sy))))
                            x2o = max(0, min(w0, int(round(x2 * sx))))
                            y2o = max(0, min(h0, int(round(y2 * sy))))

                            bw = max(1, x2o - x1o)
                            bh = max(1, y2o - y1o)
                            pad_x = int(round(bw * 0.12))
                            pad_y = int(round(bh * 0.18))
                            x1o = max(0, x1o - pad_x)
                            y1o = max(0, y1o - pad_y)
                            x2o = min(w0, x2o + pad_x)
                            y2o = min(h0, y2o + pad_y)

                            if x2o <= x1o or y2o <= y1o:
                                continue

                            roi = frame_original[y1o:y2o, x1o:x2o]
                            if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                                try:
                                    # Resize só aqui, antes do preprocess OCR.
                                    rh, rw = roi.shape[:2]
                                    m = min(rh, rw)
                                    roi_scale = 4 if m < 120 else (3 if m < 220 else 2)
                                    roi_for_ocr = _upscale_preserve_details(roi, roi_scale)

                                    img_name = f"obj_{obj_id}.jpg"
                                    ocr_queue.put_nowait((roi_for_ocr, frame_original.copy(), img_name))
                                except queue.Full:
                                    push_log("⚠️ Fila OCR cheia — descartando ROI")

            except Exception as e:
                push_log(f"⚠️ Erro YOLO/track: {e}")
                annotated = frame_proc

        # contador de vídeo na imagem
        cv2.putText(
            annotated,
            f"Total (video): {total_video}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    push_log("✅ Painel UHT inicializado")
    open_video_from_start()
    app.run(host="0.0.0.0", port=5000, debug=True)
