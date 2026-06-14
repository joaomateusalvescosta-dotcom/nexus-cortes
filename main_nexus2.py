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

from subtitle_burner import aplicar_legenda

load_dotenv()

# =============================================================================
# LOGGING ESTRUTURADO
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
MIN_SCORE_SENTIMENTO   = 0.70
MIN_SCORE_RELEVANCIA   = 0.70
MIN_DURACAO            = 15.0
MAX_DURACAO            = 90.0
MAX_GAP_MERGE          = 2.0
BUFFER_TEMPO           = 1.0
MARGEM_KEYFRAME        = 3.0

SCORE_MINIMO_VALIDACAO = 18

TEMAS_RELEVANTES = ["conselho", "polêmica", "insight valioso", "história curiosa", "piada"]

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
device_gpu           = "cuda" if torch.cuda.is_available() else "cpu"
id_gpu_transformers  = -1   # CPU — evita OOM com GPU de 4GB


# =============================================================================
# PASTAS DE TRABALHO
# =============================================================================
PASTA_AUDIOS_TEMP = "audios_temp"
PASTA_SAIDA       = "cortes_finalizados"
PASTA_TRANSCRIPTS = "transcripts"
PASTA_VIDEOS      = "videos_brutos"

os.makedirs(PASTA_AUDIOS_TEMP, exist_ok=True)
os.makedirs(PASTA_SAIDA, exist_ok=True)
os.makedirs(PASTA_TRANSCRIPTS, exist_ok=True)
os.makedirs(PASTA_VIDEOS, exist_ok=True)


# =============================================================================
# CARREGAMENTO DOS MODELOS
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
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "cookiesfrombrowser": ("firefox",),
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "remote_components": ["ejs:github"],
        "outtmpl": os.path.join(PASTA_AUDIOS_TEMP, "audio_alvo.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 60,
    }

    log.info("Baixando áudio...")
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url_youtube])

    if not os.path.exists(caminho_audio):
        raise FileNotFoundError("Falha ao extrair áudio — arquivo não encontrado após download.")

    return caminho_audio


# =============================================================================
# DOWNLOAD DO VÍDEO BRUTO
#
# Salva em videos_brutos/ com nome baseado no ID do vídeo.
# Se o arquivo já existir, reutiliza sem baixar novamente —
# útil quando você roda o pipeline mais de uma vez no mesmo vídeo.
# =============================================================================
def baixar_video_bruto(url_youtube: str) -> str:
    # Extrai o ID do vídeo para usar como nome de arquivo
    with yt_dlp.YoutubeDL({
        "quiet": True,
        "cookiesfrombrowser": ("firefox",),
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "remote_components": ["ejs:github"],
    }) as ydl:
        info = ydl.extract_info(url_youtube, download=False)
        video_id = info.get("id", "video")

    caminho_video = os.path.join(PASTA_VIDEOS, f"{video_id}.mp4")

    # Reutiliza se já existir E estiver íntegro (>1MB para evitar arquivo corrompido)
    if os.path.exists(caminho_video) and os.path.getsize(caminho_video) > 1024 * 1024:
        log.info(f"Vídeo já existe em videos_brutos/ — reutilizando: {caminho_video}")
        return caminho_video

    # Remove arquivo parcial/corrompido se existir
    if os.path.exists(caminho_video):
        log.warning("Arquivo anterior parece estar incompleto — removendo para baixar novamente.")
        os.remove(caminho_video)

    opcoes = {
        "format": "bestvideo*+bestaudio/best",
        "cookiesfrombrowser": ("firefox",),
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "remote_components": ["ejs:github"],
        "outtmpl": caminho_video,
        "quiet": True,
        "no_warnings": True,
        # Robustez de rede — o YouTube às vezes tem servidores instáveis (rr2.googlevideo.com etc)
        "retries": 10,                    # tentativas para o download como um todo
        "fragment_retries": 10,           # tentativas para cada fragmento de stream
        "socket_timeout": 60,             # 60s antes de declarar timeout (padrão é 20s)
        "retry_sleep_functions": {        # backoff exponencial entre tentativas
            "http": lambda n: min(2 ** n, 30),
            "fragment": lambda n: min(2 ** n, 30),
        },
    }

    MAX_TENTATIVAS_DOWNLOAD = 3
    ultima_excecao = None

    for tentativa in range(1, MAX_TENTATIVAS_DOWNLOAD + 1):
        try:
            log.info(f"Baixando vídeo completo [{tentativa}/{MAX_TENTATIVAS_DOWNLOAD}] para videos_brutos/...")
            with yt_dlp.YoutubeDL(opcoes) as ydl:
                ydl.download([url_youtube])

            if os.path.exists(caminho_video) and os.path.getsize(caminho_video) > 1024 * 1024:
                log.info(f"Vídeo salvo: {caminho_video}")
                return caminho_video

            raise FileNotFoundError("Arquivo não encontrado ou tamanho suspeito após download.")

        except Exception as e:
            ultima_excecao = e
            log.warning(f"Erro na tentativa {tentativa}: {e}")

            # Limpa arquivo parcial entre tentativas
            if os.path.exists(caminho_video):
                os.remove(caminho_video)

            if tentativa < MAX_TENTATIVAS_DOWNLOAD:
                espera = 15 * tentativa
                log.info(f"Aguardando {espera}s antes de tentar novamente...")
                time.sleep(espera)

    raise RuntimeError(
        f"Falha ao baixar vídeo após {MAX_TENTATIVAS_DOWNLOAD} tentativas. "
        f"Último erro: {ultima_excecao}"
    )


# =============================================================================
# CORTE LOCAL
#
# Recebe o vídeo já baixado em videos_brutos/ e usa o ffmpeg para
# extrair o trecho com precisão. Muito mais estável do que cortar
# direto no yt-dlp (que gerava "no stream" em muitos formatos).
#
# -ss antes de -i: seek rápido (nem sempre frame-accurate)
# -t: duração do trecho
# Filtro vertical: converte para 9:16 com fundo borrado
# =============================================================================
def cortar_video_local(caminho_video: str, inicio: float, fim: float, nome_arquivo: str) -> bool:
    caminho_final = os.path.join(PASTA_SAIDA, nome_arquivo)
    duracao_corte = fim - inicio

    filtro_vertical = (
        "[0:v]split[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg_rows];"
        "[fg]scale=1080:-2[fg_scaled];"
        "[bg_rows][fg_scaled]overlay=(W-w)/2:(H-h)/2"
    )

    comando_ffmpeg = [
        "ffmpeg", "-y",
        "-ss", str(inicio),
        "-i", caminho_video,
        "-t", str(duracao_corte),
        "-vf", filtro_vertical,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        caminho_final
    ]

    log.info(f"  Cortando {inicio:.1f}s → {fim:.1f}s → {nome_arquivo}")
    resultado = subprocess.run(comando_ffmpeg, capture_output=True, text=True)

    if resultado.returncode != 0:
        log.error(f"  ffmpeg falhou:\n{resultado.stderr[-400:]}")
        return False

    log.info(f"  Corte salvo: {caminho_final}")
    return True


# =============================================================================
# ANÁLISE DE SEGMENTOS
# =============================================================================
def segmento_e_interessante(texto: str) -> bool:
    if len(texto.strip()) < 10:
        return False

    texto_lower = texto.lower()

    marcadores_risada = ["[risos]", "(risos)", "haha", "kkk", "rsrs", "hehe"]
    if any(m in texto_lower for m in marcadores_risada):
        return True

    resultado_sentimento = modelo_sentimento(texto[:512])[0]
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
# AJUSTE DE FIM DE CORTE — BASEADO EM PAUSAS REAIS DO ÁUDIO
#
# Por que pausa real (gap entre segmentos) e não só pontuação?
#   O Whisper transcreve pontuação de forma inconsistente em PT-BR (esquece
#   pontos, troca por vírgula, etc). Pausa real é um sinal acústico que não
#   depende do Whisper acertar a transcrição — se houve silêncio entre dois
#   segmentos, é porque o falante de fato parou de falar.
#
# Como funciona:
#   1. Procura pausas (gap entre segmentos consecutivos) na janela ao redor
#      do fim calculado
#   2. Filtra só pausas longas o suficiente para serem fim de fala (não vírgula)
#   3. Pega a pausa mais próxima do fim calculado
#   4. Adapta o "respiro" depois da pausa conforme o tipo de finalização:
#        - Risada/exclamação → 2s para "digerir" o momento
#        - Ponto final → 0.3s, corte objetivo
#        - Sem pontuação → 0.5s, transição natural
#   5. Se não achar pausa, mantém o fim original (fallback seguro)
#
# Valores baseados em pesquisas específicas em Português Brasileiro:
#   - Fronteira entonacional (IPh) em PT-BR: ~300ms (Fortunato-Tavares, CoDAS 2023)
#   - Pausa mínima perceptível em PT-BR: ~100ms (Mello et al., De Gruyter)
#   - Falantes BR usam gaps mais largos antes de turno (ResearchGate 2024)
# =============================================================================
JANELA_AJUSTE_FIM       = 8.0   # tolerância para buscar pausa (segundos)
PAUSA_MINIMA_FIM_FALA   = 0.5   # gap mínimo para considerar fim de fala (acima dos 300ms de IPh)
RESPIRO_RISADA          = 2.0   # respiro após risada/exclamação
RESPIRO_PONTO_FINAL     = 0.3   # respiro após ponto final
RESPIRO_PAUSA_NEUTRA    = 0.5   # respiro após pausa sem pontuação clara

MARCADORES_ALTA_ENERGIA = ("?", "!", "kkk", "kkkk", "haha", "rsrs", "[risos]", "(risos)")
PONTUACAO_FINAL         = (".", "...", "…")


def _classificar_finalizacao(texto: str) -> tuple[str, float]:
    """
    Analisa o texto final de um segmento e retorna o tipo de finalização
    e o respiro apropriado em segundos.
    """
    texto_limpo = texto.strip().lower()

    if any(texto_limpo.endswith(m) for m in MARCADORES_ALTA_ENERGIA):
        return "risada/exclamação", RESPIRO_RISADA

    # Risada pode aparecer no meio do texto, não só no final
    if any(m in texto_limpo for m in ("kkk", "haha", "[risos]", "(risos)")):
        return "risada/exclamação", RESPIRO_RISADA

    if any(texto_limpo.endswith(p) for p in PONTUACAO_FINAL):
        return "ponto final", RESPIRO_PONTO_FINAL

    return "pausa neutra", RESPIRO_PAUSA_NEUTRA


def ajustar_fim_corte(fim_calculado: float, segmentos_whisper: list, duracao_total: float) -> float:
    """
    Encontra um ponto de término natural usando pausas reais do áudio.
    Veja docstring acima da função para detalhes da estratégia.
    """
    janela_min = fim_calculado - JANELA_AJUSTE_FIM
    janela_max = fim_calculado + JANELA_AJUSTE_FIM

    # Identifica pausas entre segmentos consecutivos dentro da janela
    # Cada pausa é um candidato a "fim de fala"
    pausas_candidatas = []
    for i in range(len(segmentos_whisper) - 1):
        seg_atual    = segmentos_whisper[i]
        seg_proximo  = segmentos_whisper[i + 1]
        gap          = seg_proximo["start"] - seg_atual["end"]

        # Só considera pausas longas o suficiente e dentro da janela
        if gap >= PAUSA_MINIMA_FIM_FALA and janela_min <= seg_atual["end"] <= janela_max:
            pausas_candidatas.append({
                "fim_segmento": seg_atual["end"],
                "texto":        seg_atual["text"],
                "gap":          gap,
            })

    if not pausas_candidatas:
        # Nenhuma pausa real encontrada — mantém o fim original (fallback seguro)
        log.info(f"    Fim mantido: {fim_calculado:.1f}s (nenhuma pausa real na janela)")
        return min(fim_calculado, duracao_total)

    # Escolhe a pausa mais próxima do fim calculado
    melhor = min(pausas_candidatas, key=lambda p: abs(p["fim_segmento"] - fim_calculado))

    # Classifica o tipo de finalização e aplica o respiro apropriado
    tipo, respiro = _classificar_finalizacao(melhor["texto"])
    fim_ajustado  = melhor["fim_segmento"] + respiro

    log.info(
        f"    Fim ajustado: {fim_calculado:.1f}s → {fim_ajustado:.1f}s "
        f"({tipo}, pausa de {melhor['gap']:.2f}s + respiro de {respiro}s)"
    )
    return min(fim_ajustado, duracao_total)


# =============================================================================
# VALIDAÇÃO AUTOMÁTICA DE ENGAJAMENTO
#
# Score de 0–100 composto por três camadas:
#
# 1. TÉCNICA (30 pts)
#    - Duração ideal 30–60s: +15 pts
#    - Silêncio no início: até -10 pts
#    - Densidade de fala (wpm): 0–5 pts
#
# 2. CONTEÚDO (40 pts)
#    - Gancho nos primeiros 30s: +20 pts
#    - Virada narrativa: +20 pts
#
# 3. EMOÇÃO (30 pts)
#    - Intensidade máxima de sentimento: 0–15 pts
#    - Variação emocional ao longo do trecho: 0–15 pts
# =============================================================================
def calcular_score_engajamento(
    trecho: dict,
    segmentos_whisper: list,
    duracao_total: float
) -> dict:

    inicio  = trecho["start"]
    fim     = trecho["end"]
    duracao = fim - inicio

    segs_do_trecho = [
        s for s in segmentos_whisper
        if s["end"] >= inicio and s["start"] <= fim
    ]

    transcript_completo = " ".join(s["text"].strip() for s in segs_do_trecho)
    transcript_lower    = transcript_completo.lower()
    palavras            = transcript_completo.split()

    score    = 0
    detalhes = {}

    # CAMADA 1 — TÉCNICA
    if 30 <= duracao <= 60:
        pts_duracao = 15
    elif duracao < 30:
        pts_duracao = max(0, int(duracao / 2))
    else:
        pts_duracao = max(0, 15 - int((duracao - 60) / 3))

    silencio_inicio = 0.0
    if segs_do_trecho:
        silencio_inicio = max(0, segs_do_trecho[0]["start"] - inicio)
    pts_silencio = max(0, 10 - int(silencio_inicio * 10))

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

    # CAMADA 2 — CONTEÚDO
    segs_inicio  = [s for s in segs_do_trecho if s["start"] - inicio <= 30]
    texto_inicio = " ".join(s["text"].lower() for s in segs_inicio)
    tem_gancho   = any(p in texto_inicio for p in PALAVRAS_GANCHO)
    pts_gancho   = 20 if tem_gancho else 0

    tem_virada = any(p in transcript_lower for p in PALAVRAS_VIRADA)
    pts_virada = 20 if tem_virada else 0

    score += pts_gancho + pts_virada
    detalhes["conteudo"] = {
        "tem_gancho": tem_gancho,
        "tem_virada": tem_virada,
        "pontos": pts_gancho + pts_virada
    }

    # CAMADA 3 — EMOÇÃO
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
        pts_intensidade      = int(score_max_sentimento * 15)
        pts_variacao         = min(15, int(variacao_sentimento * 60))
    else:
        pts_intensidade      = 0
        pts_variacao         = 0
        score_max_sentimento = 0
        variacao_sentimento  = 0

    score += pts_intensidade + pts_variacao
    detalhes["emocao"] = {
        "score_max": round(score_max_sentimento, 3),
        "variacao": round(variacao_sentimento, 3),
        "pontos": pts_intensidade + pts_variacao
    }

    score_final            = min(100, score)
    detalhes["score_total"] = score_final
    detalhes["aprovado"]    = score_final >= SCORE_MINIMO_VALIDACAO

    return detalhes


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def processar_url(url: str):
    log.info("=" * 50)
    log.info("INICIANDO PIPELINE DE CORTES AUTÔNOMOS")
    log.info("=" * 50)

    # 1. Extrai áudio para transcrição
    caminho_audio = extrair_audio(url)

    # 2. Transcrição com Whisper
    #    word_timestamps=True faz o Whisper retornar timestamp de cada palavra
    #    individualmente, necessário para legendas word-by-word.
    #    Cache: se já transcrevemos esse vídeo antes, reutiliza.
    cache_transcricao = os.path.join(PASTA_TRANSCRIPTS, "_cache_transcricao.json")
    resultado          = None

    if os.path.exists(cache_transcricao):
        try:
            with open(cache_transcricao, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            # Cache válido só se for do mesmo áudio E tiver word_timestamps
            if (cache_data.get("audio_size") == os.path.getsize(caminho_audio)
                    and cache_data.get("tem_words", False)):
                log.info("Transcrição em cache encontrada — reutilizando.")
                resultado = {"segments": cache_data["segments"]}
        except Exception as e:
            log.warning(f"Cache inválido, refazendo transcrição: {e}")

    if resultado is None:
        log.info("Transcrevendo áudio com Whisper (small) + word_timestamps...")
        resultado = modelo_whisper.transcribe(
            caminho_audio,
            fp16=(device_gpu == "cuda"),
            word_timestamps=True,   # necessário para legendas word-by-word
        )

        # Salva cache para reutilizar caso o pipeline falhe depois
        try:
            with open(cache_transcricao, "w", encoding="utf-8") as f:
                json.dump({
                    "audio_size": os.path.getsize(caminho_audio),
                    "segments":   resultado["segments"],
                    "tem_words":  True,
                }, f, ensure_ascii=False)
            log.info("Cache de transcrição salvo.")
        except Exception as e:
            log.warning(f"Não foi possível salvar cache: {e}")

    segmentos_whisper = resultado.get("segments", [])

    # Extrai todas as palavras com timestamps em uma lista plana
    # Cada palavra tem: {word, start, end, probability}
    palavras_whisper = []
    for seg in segmentos_whisper:
        if "words" in seg:
            palavras_whisper.extend(seg["words"])

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

    # 4. Mescla segmentos próximos
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
            trecho["validacao"]  = validacao
            trecho["nome_saida"] = f"corte_{i:02d}_score{validacao['score_total']}.mp4"
            cortes_aprovados.append(trecho)

            # Salva contexto para o publish.py gerar metadados sem re-transcrever
            segs_do_trecho    = [s for s in segmentos_whisper if s["end"] >= trecho["start"] and s["start"] <= trecho["end"]]
            transcript_trecho = " ".join(s["text"].strip() for s in segs_do_trecho)

            temas_trecho = []
            if transcript_trecho.strip():
                try:
                    res_temas    = modelo_relevancia(transcript_trecho[:512], candidate_labels=TEMAS_RELEVANTES)
                    temas_trecho = [
                        label for label, sc in zip(res_temas["labels"], res_temas["scores"])
                        if sc > 0.20
                    ][:3]
                except Exception:
                    pass

            contexto     = {"transcript": transcript_trecho, "temas": temas_trecho, "score": validacao["score_total"]}
            nome_json    = trecho["nome_saida"].replace(".mp4", ".json")
            caminho_json = os.path.join(PASTA_TRANSCRIPTS, nome_json)
            try:
                with open(caminho_json, "w", encoding="utf-8") as f:
                    json.dump(contexto, f, ensure_ascii=False, indent=2)
                log.info(f"    Contexto salvo: {caminho_json}")
            except Exception as e:
                log.warning(f"    Não foi possível salvar contexto: {e}")

    if not cortes_aprovados:
        log.info("Nenhum corte aprovado. Encerrando sem download de vídeo.")
        if os.path.exists(caminho_audio):
            os.remove(caminho_audio)
        return

    # 6. Baixa o vídeo completo UMA vez em videos_brutos/
    #    (reutiliza se já existir — não baixa de novo)
    caminho_video = baixar_video_bruto(url)

    # 7. Gera todos os cortes localmente em paralelo
    log.info(f"\n{len(cortes_aprovados)} corte(s) aprovado(s). Gerando cortes...\n")

    def processar_corte(trecho):
        inicio_seguro = max(0, trecho["start"] - MARGEM_KEYFRAME)

        # Ajusta o fim para coincidir com término natural de fala no Whisper
        fim_ajustado  = ajustar_fim_corte(trecho["end"], segmentos_whisper, duracao_total)
        fim_seguro    = min(duracao_total, fim_ajustado + MARGEM_KEYFRAME)

        # 1. Corta o vídeo (sem legenda ainda)
        nome_temp        = trecho["nome_saida"].replace(".mp4", "_temp.mp4")
        sucesso_corte    = cortar_video_local(caminho_video, inicio_seguro, fim_seguro, nome_temp)

        if not sucesso_corte:
            return False

        caminho_corte_temp = os.path.join(PASTA_SAIDA, nome_temp)
        caminho_corte_final = os.path.join(PASTA_SAIDA, trecho["nome_saida"])

        # 2. Aplica legenda word-by-word
        sucesso_legenda = aplicar_legenda(
            video_entrada = caminho_corte_temp,
            palavras      = palavras_whisper,
            inicio_corte  = inicio_seguro,
            fim_corte     = fim_seguro,
            video_saida   = caminho_corte_final,
        )

        # 3. Remove a versão sem legenda
        if sucesso_legenda and os.path.exists(caminho_corte_temp):
            os.remove(caminho_corte_temp)
        elif not sucesso_legenda:
            # Se falhar a legenda, renomeia o temp para o nome final
            # (melhor ter o vídeo sem legenda do que perder o trabalho)
            log.warning(f"  Legenda falhou para {trecho['nome_saida']} — salvando sem legenda")
            if os.path.exists(caminho_corte_temp):
                os.rename(caminho_corte_temp, caminho_corte_final)

        return True

    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = {executor.submit(processar_corte, t): t for t in cortes_aprovados}
        for futuro in as_completed(futuros):
            trecho = futuros[futuro]
            try:
                futuro.result()
            except Exception as e:
                log.error(f"Erro ao processar {trecho['nome_saida']}: {e}")

    # 8. Limpeza — remove só o áudio temp, vídeo bruto fica em videos_brutos/
    if os.path.exists(caminho_audio):
        os.remove(caminho_audio)

    log.info(f"\nOperação concluída. Cortes salvos em: {PASTA_SAIDA}/")
    log.info(f"Vídeo bruto mantido em: {PASTA_VIDEOS}/ (delete manualmente se não precisar mais)")


# =============================================================================
# ENTRADA
# =============================================================================
if __name__ == "__main__":
    url_alvo = input("URL do vídeo ou live (YouTube): ").strip()

    if url_alvo:
        processar_url(url_alvo)
    else:
        log.error("URL inválida. Encerrando.")
