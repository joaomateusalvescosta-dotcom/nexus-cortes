import os
import json
import time
import logging
import statistics
import torch
import whisper
import yt_dlp
from transformers import pipeline
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# LOGGING ESTRUTURADO
# Substitui print() por logging para facilitar debug e futura integração
# com arquivos de log ou ferramentas de monitoramento
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# =============================================================================
# PARÂMETROS DE CALIBRAÇÃO
# =============================================================================
MIN_SCORE_SENTIMENTO  = 0.70   # Reduzido levemente pois o XLM-RoBERTa é mais preciso
MIN_SCORE_RELEVANCIA  = 0.70
MIN_DURACAO           = 15.0
MAX_DURACAO           = 90.0
MAX_GAP_MERGE         = 2.0
BUFFER_TEMPO          = 1.0
MARGEM_KEYFRAME       = 3.0

SCORE_MINIMO_VALIDACAO = 18    # Score de 0-100 para aprovar o corte automaticamente

TEMAS_RELEVANTES = ["conselho", "polêmica", "insight valioso", "história curiosa", "piada"]

# Heurísticas de engajamento baseadas em estudos de plataformas de short-form video
PALAVRAS_GANCHO = [
    "você sabia", "nunca ouvi", "descobri", "a verdade", "segredo",
    "erro", "aprendi", "mudou minha", "por que", "como eu", "história",
    "inacreditável", "surpreendente", "confissão", "revelar"
]
PALAVRAS_VIRADA = [
    "mas", "porém", "entretanto", "na verdade", "o problema é",
    "o que ninguém fala", "acontece que", "até que", "foi quando",
    "aí que", "o detalhe é", "pior que", "melhor que",
    "aí", "então", "só que", "cara", "olha", "escuta",
    "deixa eu te falar", "sabe o que é", "o negócio é",
    "e aí", "mas aí", "foi aí", "e foi", "até que enfim"
]


# =============================================================================
# HARDWARE
# =============================================================================
device_gpu            = "cuda" if torch.cuda.is_available() else "cpu"
device_cpu            = "cpu"
id_gpu_transformers   = -1


# =============================================================================
# PASTAS DE TRABALHO
# =============================================================================
PASTA_AUDIOS_TEMP = "audios_temp"
PASTA_SAIDA       = "cortes_finalizados"
PASTA_TRANSCRIPTS = "transcripts"

os.makedirs(PASTA_AUDIOS_TEMP, exist_ok=True)
os.makedirs(PASTA_SAIDA, exist_ok=True)
os.makedirs(PASTA_TRANSCRIPTS, exist_ok=True)


# =============================================================================
# CARREGAMENTO DOS MODELOS
#
# Whisper: transcrição de áudio
#   → Trocamos "tiny" por "small" para melhor qualidade de transcrição.
#     "tiny" é mais rápido mas erra muito em fala informal/PT-BR, o que
#     compromete toda a análise downstream.
#
# XLM-RoBERTa: classificação de sentimento multilingual
#   → Substitui o BERTweet-PT, que foi treinado apenas em tweets.
#     O XLM-RoBERTa foi treinado em 100 idiomas e lida melhor com
#     fala espontânea de podcasts/entrevistas.
#
# DistilBART: classificação zero-shot de relevância temática
#   → Mantido pois não há alternativa PT-BR equivalente para zero-shot.
# =============================================================================
log.info("Carregando modelos de IA...")

modelo_whisper = whisper.load_model("small", device=device_gpu)

modelo_sentimento = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=id_gpu_transformers
)

modelo_relevancia = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=id_gpu_transformers
)

log.info("Modelos carregados.\n")


# =============================================================================
# EXTRAÇÃO DE ÁUDIO
# =============================================================================
def extrair_audio(url_youtube: str) -> str:
    caminho_audio = os.path.join(PASTA_AUDIOS_TEMP, "audio_alvo.mp3")

    if os.path.exists(caminho_audio):
        os.remove(caminho_audio)

    opcoes = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(PASTA_AUDIOS_TEMP, "audio_alvo.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    log.info("Baixando áudio...")
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url_youtube])

    if not os.path.exists(caminho_audio):
        raise FileNotFoundError("Falha ao extrair áudio — arquivo não encontrado após download.")

    return caminho_audio


# =============================================================================
# CORTE E RENDERIZAÇÃO DO VÍDEO
# Retry automático: tenta até 3 vezes com backoff de 5s entre tentativas
# ffmpeg: captura stderr para log estruturado, não silencia erros
# =============================================================================
def cortar_video(url_youtube: str, inicio: float, fim: float, nome_arquivo: str) -> bool:
    caminho_final = os.path.join(PASTA_SAIDA, nome_arquivo)
    caminho_temp  = os.path.join(PASTA_SAIDA, f"temp_{nome_arquivo}")

    MAX_TENTATIVAS = 3

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            opcoes_download = {
                "format": "b[ext=mp4]/b/best",
                "outtmpl": caminho_temp,
                "download_ranges": lambda info_dict, ydl: [{"start_time": inicio, "end_time": fim}],
                "force_keyframes_at_cuts": True,
                "quiet": True,
                "no_warnings": True,
            }

            log.info(f"  [tentativa {tentativa}/{MAX_TENTATIVAS}] Baixando {inicio:.1f}s → {fim:.1f}s...")
            with yt_dlp.YoutubeDL(opcoes_download) as ydl:
                ydl.download([url_youtube])

            if not os.path.exists(caminho_temp):
                raise FileNotFoundError("Arquivo temporário não foi criado pelo yt-dlp.")

            filtro_vertical = (
                "[0:v]split[bg][fg];"
                "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg_rows];"
                "[fg]scale=1080:-2[fg_scaled];"
                "[bg_rows][fg_scaled]overlay=(W-w)/2:(H-h)/2"
            )

            comando_ffmpeg = [
                "ffmpeg", "-y", "-i", caminho_temp,
                "-vf", filtro_vertical,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                caminho_final
            ]

            # capture_output=True evita flood no terminal e permite inspecionar erros
            resultado_ffmpeg = subprocess.run(
                comando_ffmpeg,
                capture_output=True,
                text=True
            )

            if resultado_ffmpeg.returncode != 0:
                raise RuntimeError(f"ffmpeg falhou:\n{resultado_ffmpeg.stderr[-500:]}")

            log.info(f"  Corte salvo: {caminho_final}")
            return True

        except Exception as e:
            log.warning(f"  Erro na tentativa {tentativa}: {e}")
            if tentativa < MAX_TENTATIVAS:
                log.info(f"  Aguardando 5s antes de tentar novamente...")
                time.sleep(5)
        finally:
            if os.path.exists(caminho_temp):
                os.remove(caminho_temp)

    log.error(f"  Falha definitiva ao gerar {nome_arquivo} após {MAX_TENTATIVAS} tentativas.")
    return False


# =============================================================================
# ANÁLISE DE SEGMENTOS
# =============================================================================
def segmento_e_interessante(texto: str) -> bool:
    if len(texto.strip()) < 10:
        return False

    texto_lower = texto.lower()

    # Detecção de risada — limitação conhecida: "kkk" pode gerar falsos positivos
    marcadores_risada = ["[risos]", "(risos)", "haha", "kkk", "rsrs", "hehe"]
    if any(m in texto_lower for m in marcadores_risada):
        return True

    resultado_sentimento = modelo_sentimento(texto[:512])[0]  # XLM-RoBERTa aceita até 512 tokens
    if resultado_sentimento["score"] > MIN_SCORE_SENTIMENTO:
        return True

    resultado_relevancia = modelo_relevancia(texto, candidate_labels=TEMAS_RELEVANTES)
    if max(resultado_relevancia["scores"]) > MIN_SCORE_RELEVANCIA:
        return True

    return False


# =============================================================================
# MESCLAGEM DE SEGMENTOS PRÓXIMOS
# =============================================================================
def mesclar_segmentos(segmentos: list) -> list:
    if not segmentos:
        return []

    resultado = []
    atual = segmentos[0].copy()

    for proximo in segmentos[1:]:
        gap = proximo["start"] - atual["end"]
        duracao_mesclada = proximo["end"] - atual["start"]

        if gap < MAX_GAP_MERGE and duracao_mesclada <= MAX_DURACAO:
            atual["end"] = max(atual["end"], proximo["end"])
        else:
            if atual["end"] - atual["start"] >= MIN_DURACAO:
                resultado.append(atual)
            atual = proximo.copy()

    if atual["end"] - atual["start"] >= MIN_DURACAO:
        resultado.append(atual)

    return resultado


# =============================================================================
# VALIDAÇÃO AUTOMÁTICA DE ENGAJAMENTO
#
# Score de 0–100 composto por três camadas:
#
# 1. TÉCNICA (30 pts): avalia qualidade objetiva do corte
#    - Duração ideal para short-form (30–60s): +15 pts
#    - Silêncio no início/fim: -10 pts por segundo acima de 1s
#    - Densidade de fala (palavras/minuto): escala de 0–15
#
# 2. CONTEÚDO (40 pts): avalia qualidade narrativa via transcript
#    - Gancho nos primeiros 30s: +20 pts
#    - Presença de "virada" ou contraste narrativo: +20 pts
#
# 3. EMOÇÃO (30 pts): avalia expressividade emocional
#    - Score máximo de sentimento nos segmentos: 0–15 pts
#    - Variação de sentimento ao longo do trecho: 0–15 pts
#      (variação alta = conversa dinâmica, não monótona)
# =============================================================================
def calcular_score_engajamento(
    trecho: dict,
    segmentos_whisper: list,
    duracao_total: float
) -> dict:

    inicio = trecho["start"]
    fim    = trecho["end"]
    duracao = fim - inicio

    # Filtra os segmentos do Whisper que pertencem a este trecho
    segs_do_trecho = [
        s for s in segmentos_whisper
        if s["end"] >= inicio and s["start"] <= fim
    ]

    transcript_completo = " ".join(s["text"].strip() for s in segs_do_trecho)
    transcript_lower    = transcript_completo.lower()
    palavras            = transcript_completo.split()

    score = 0
    detalhes = {}

    # ------------------------------------------------------------------
    # CAMADA 1 — TÉCNICA (30 pts)
    # ------------------------------------------------------------------

    # Duração ideal: 30–60s rende melhor em plataformas de short-form
    if 30 <= duracao <= 60:
        pts_duracao = 15
    elif duracao < 30:
        pts_duracao = max(0, int(duracao / 2))      # proporcional abaixo de 30s
    else:
        pts_duracao = max(0, 15 - int((duracao - 60) / 3))  # perde 1pt a cada 3s acima de 60

    # Silêncio no início do trecho
    silencio_inicio = 0.0
    if segs_do_trecho:
        silencio_inicio = max(0, segs_do_trecho[0]["start"] - inicio)

    pts_silencio = max(0, 10 - int(silencio_inicio * 10))   # -10 pts por segundo de silêncio

    # Densidade de fala 
    wpm = (len(palavras) / duracao) * 60 if duracao > 0 else 0
    if 100 <= wpm <= 170:
        pts_densidade = 5
    elif wpm > 0:
        pts_densidade = max(0, 5 - int(abs(wpm - 135) / 20))
    else:
        pts_densidade = 0

    score += pts_duracao + pts_silencio + pts_densidade
    detalhes["tecnica"] = {
        "duracao_s": round(duracao, 1),
        "silencio_inicio_s": round(silencio_inicio, 2),
        "wpm": round(wpm, 0),
        "pontos": pts_duracao + pts_silencio + pts_densidade
    }

    # ------------------------------------------------------------------
    # CAMADA 2 — CONTEÚDO / NARRATIVA (40 pts)
    # ------------------------------------------------------------------

    # Gancho: palavra de gancho nos primeiros ~30s do trecho
    segs_inicio = [
        s for s in segs_do_trecho
        if s["start"] - inicio <= 30
    ]
    texto_inicio = " ".join(s["text"].lower() for s in segs_inicio)
    tem_gancho   = any(p in texto_inicio for p in PALAVRAS_GANCHO)
    pts_gancho   = 20 if tem_gancho else 0

    # Virada narrativa: contraste ou inversão no decorrer do trecho
    tem_virada  = any(p in transcript_lower for p in PALAVRAS_VIRADA)
    pts_virada  = 20 if tem_virada else 0

    score += pts_gancho + pts_virada
    detalhes["conteudo"] = {
        "tem_gancho": tem_gancho,
        "tem_virada": tem_virada,
        "pontos": pts_gancho + pts_virada
    }

    # ------------------------------------------------------------------
    # CAMADA 3 — EMOÇÃO (30 pts)
    # ------------------------------------------------------------------
    scores_sentimento = []
    for seg in segs_do_trecho:
        texto_seg = seg["text"].strip()
        if len(texto_seg) > 10:
            try:
                res = modelo_sentimento(texto_seg[:512])[0]
                scores_sentimento.append(res["score"])
            except Exception:
                pass

    if scores_sentimento:
        score_max_sentimento = max(scores_sentimento)
        variacao_sentimento  = statistics.stdev(scores_sentimento) if len(scores_sentimento) > 1 else 0

        pts_intensidade = int(score_max_sentimento * 15)
        pts_variacao    = min(15, int(variacao_sentimento * 60))   # stdev de 0–0.25 mapeado para 0–15
    else:
        pts_intensidade = 0
        pts_variacao    = 0
        score_max_sentimento = 0
        variacao_sentimento  = 0

    score += pts_intensidade + pts_variacao
    detalhes["emocao"] = {
        "score_max": round(score_max_sentimento, 3),
        "variacao": round(variacao_sentimento, 3),
        "pontos": pts_intensidade + pts_variacao
    }

    # ------------------------------------------------------------------
    # RESULTADO FINAL
    # ------------------------------------------------------------------
    score_final   = min(100, score)
    aprovado      = score_final >= SCORE_MINIMO_VALIDACAO
    detalhes["score_total"] = score_final
    detalhes["aprovado"]    = aprovado

    return detalhes


# =============================================================================
# PIPELINE PRINCIPAL
# Downloads dos cortes rodam em paralelo via ThreadPoolExecutor
# =============================================================================
def processar_url(url: str):
    log.info("=" * 50)
    log.info("INICIANDO PIPELINE DE CORTES AUTÔNOMOS")
    log.info("=" * 50)

    # 1. Áudio
    caminho_audio = extrair_audio(url)

    # 2. Transcrição
    log.info("Transcrevendo áudio com Whisper (small)...")
    resultado = modelo_whisper.transcribe(caminho_audio, fp16=(device_gpu == "cuda"))

    segmentos_whisper = resultado.get("segments", [])

    if not segmentos_whisper:
        log.error("Whisper não retornou segmentos. Verifique o áudio.")
        return

    duracao_total = segmentos_whisper[-1]["end"]

    # 3. Identifica segmentos interessantes
    log.info("Analisando segmentos por sentimento e relevância...")
    pontos_interesse = []

    for seg in segmentos_whisper:
        texto = seg["text"].strip()
        if segmento_e_interessante(texto):
            pontos_interesse.append({
                "start": max(0, seg["start"] - BUFFER_TEMPO),
                "end":   min(duracao_total, seg["end"] + BUFFER_TEMPO),
            })

    # 4. Mescla
    cortes_candidatos = mesclar_segmentos(pontos_interesse)

    # 5. Validação de engajamento
    log.info(f"\nValidando {len(cortes_candidatos)} candidatos a corte...")
    cortes_aprovados = []

    for i, trecho in enumerate(cortes_candidatos, 1):
        validacao = calcular_score_engajamento(trecho, segmentos_whisper, duracao_total)
        status    = "✓ APROVADO" if validacao["aprovado"] else "✗ REPROVADO"

        log.info(
            f"  Corte {i:02d} [{trecho['start']:.1f}s–{trecho['end']:.1f}s] "
            f"Score: {validacao['score_total']}/100 — {status}"
        )
        log.info(
            f"    Técnica: {validacao['tecnica']['pontos']}pts | "
            f"Conteúdo: {validacao['conteudo']['pontos']}pts | "
            f"Emoção: {validacao['emocao']['pontos']}pts"
        )

        if validacao["aprovado"]:
            trecho["validacao"]   = validacao
            trecho["nome_saida"]  = f"corte_{i:02d}_score{validacao['score_total']}.mp4"
            cortes_aprovados.append(trecho)

            # Salva contexto para o publish.py gerar metadados sem re-transcrever
            segs_do_trecho = [
                s for s in segmentos_whisper
                if s["end"] >= trecho["start"] and s["start"] <= trecho["end"]
            ]
            transcript_trecho = " ".join(s["text"].strip() for s in segs_do_trecho)

            # Detecta temas predominantes no trecho para o gerador de hashtags
            temas_trecho = []
            if transcript_trecho.strip():
                try:
                    res_temas = modelo_relevancia(transcript_trecho[:512], candidate_labels=TEMAS_RELEVANTES)
                    temas_trecho = [
                        label for label, score in zip(res_temas["labels"], res_temas["scores"])
                        if score > 0.20
                    ][:3]
                except Exception:
                    pass

            contexto = {
                "transcript": transcript_trecho,
                "temas":      temas_trecho,
                "score":      validacao["score_total"],
            }
            nome_json = trecho["nome_saida"].replace(".mp4", ".json")
            caminho_json = os.path.join(PASTA_TRANSCRIPTS, nome_json)
            try:
                with open(caminho_json, "w", encoding="utf-8") as f:
                    json.dump(contexto, f, ensure_ascii=False, indent=2)
                log.info(f"    Contexto salvo: {caminho_json}")
            except Exception as e:
                log.warning(f"    Não foi possível salvar contexto: {e}")

    log.info(f"\n{len(cortes_aprovados)}/{len(cortes_candidatos)} cortes aprovados. Iniciando downloads em paralelo...\n")

    # 6. Downloads paralelos
    def processar_corte(trecho):
        inicio_seguro = max(0, trecho["start"] - MARGEM_KEYFRAME)
        fim_seguro    = min(duracao_total, trecho["end"] + MARGEM_KEYFRAME)
        return cortar_video(url, inicio_seguro, fim_seguro, trecho["nome_saida"])

    # max_workers=3: evita sobrecarregar a conexão com muitos downloads simultâneos
    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = {executor.submit(processar_corte, t): t for t in cortes_aprovados}
        for futuro in as_completed(futuros):
            trecho = futuros[futuro]
            try:
                futuro.result()
            except Exception as e:
                log.error(f"Erro ao processar {trecho['nome_saida']}: {e}")

    # 7. Limpeza
    if os.path.exists(caminho_audio):
        os.remove(caminho_audio)

    log.info(f"\nOperação concluída. Arquivos salvos em: {PASTA_SAIDA}/")


# =============================================================================
# ENTRADA
# =============================================================================
if __name__ == "__main__":
    url_alvo = input("URL do vídeo ou live (YouTube): ").strip()

    if url_alvo:
        processar_url(url_alvo)
    else:
        log.error("URL inválida. Encerrando.")
