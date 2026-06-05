import os
import torch
import whisper
import yt_dlp
from transformers import pipeline
import subprocess

# =============================================================================
# PARÂMETROS DE CALIBRAÇÃO
# Ajuste esses valores para controlar o comportamento dos cortes
# =============================================================================
MIN_SCORE_SENTIMENTO = 0.90   # Score mínimo de emoção para considerar o trecho
MIN_SCORE_RELEVANCIA = 0.90   # Score mínimo de relevância temática
MIN_DURACAO = 15.0            # Duração mínima de um corte (segundos)
MAX_DURACAO = 90.0            # Duração máxima de um corte (segundos)
MAX_GAP_MERGE = 2.0           # Gap máximo entre trechos para mesclar (segundos)
BUFFER_TEMPO = 1.0            # Margem de tempo antes/depois de cada trecho (segundos)
MARGEM_KEYFRAME = 3.0         # Margem extra para garantir keyframe limpo no ffmpeg

TEMAS_RELEVANTES = ["conselho", "polêmica", "insight valioso", "história curiosa", "piada"]

# =============================================================================
# HARDWARE
# =============================================================================
device_gpu = "cuda" if torch.cuda.is_available() else "cpu"
device_cpu = "cpu"
id_gpu_transformers = 0 if torch.cuda.is_available() else -1

# =============================================================================
# PASTAS DE TRABALHO
# =============================================================================
PASTA_AUDIOS_TEMP = "audios_temp"
PASTA_SAIDA = "cortes_finalizados"

os.makedirs(PASTA_AUDIOS_TEMP, exist_ok=True)
os.makedirs(PASTA_SAIDA, exist_ok=True)

# =============================================================================
# CARREGAMENTO DOS MODELOS
# Whisper: transcrição de áudio
# BERTweet: classificação de sentimento em PT-BR
# DistilBART: classificação zero-shot de relevância temática
# =============================================================================
print("Carregando modelos de IA...")

modelo_whisper = whisper.load_model("tiny", device=device_gpu)

modelo_sentimento = pipeline(
    "text-classification",
    model="pysentimiento/bertweet-pt-sentiment",
    device=id_gpu_transformers
)

modelo_relevancia = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=id_gpu_transformers
)

print("Modelos carregados.\n")


# =============================================================================
# EXTRAÇÃO DE ÁUDIO
# Baixa apenas o áudio do vídeo para transcrição — evita baixar o vídeo inteiro
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

    print("Baixando áudio...")
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url_youtube])

    return caminho_audio


# =============================================================================
# CORTE E RENDERIZAÇÃO DO VÍDEO
# Baixa apenas o trecho necessário e aplica fundo borrado para formato vertical
# =============================================================================
def cortar_video(url_youtube: str, inicio: float, fim: float, nome_arquivo: str):
    caminho_final = os.path.join(PASTA_SAIDA, nome_arquivo)
    caminho_temp = os.path.join(PASTA_SAIDA, f"temp_{nome_arquivo}")

    # Baixa apenas o intervalo do vídeo 
    opcoes_download = {
        "format": "b[ext=mp4]/b/best",
        "outtmpl": caminho_temp,
        "download_ranges": lambda info_dict, ydl: [{"start_time": inicio, "end_time": fim}],
        "force_keyframes_at_cuts": True,  
        "quiet": True,
        "no_warnings": True,
    }

    print(f"  Baixando trecho {inicio:.1f}s → {fim:.1f}s...")
    try:
        with yt_dlp.YoutubeDL(opcoes_download) as ydl:
            ydl.download([url_youtube])
    except Exception as e:
        print(f"  Erro no download: {e}")
        return

    # Filtro ffmpeg: converte para formato vertical 9:16 com fundo borrado
    # Split: [bg] recebe o fundo borrado, [fg] recebe o vídeo original centralizado
    filtro_vertical = (
        "[0:v]split[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg_rows];"
        "[fg]scale=1080:-2[fg_scaled];"
        "[bg_rows][fg_scaled]overlay=(W-w)/2:(H-h)/2"
    )

    comando_ffmpeg = [
        "ffmpeg", "-y", "-i", caminho_temp,
        "-vf", filtro_vertical,
        "-c:v", "libx264", "-preset", "ultrafast",  # ultrafast: prioriza velocidade
        "-c:a", "aac",
        caminho_final
    ]

    subprocess.run(comando_ffmpeg)

    if os.path.exists(caminho_temp):
        os.remove(caminho_temp)

    print(f"  Corte salvo: {caminho_final}")


# =============================================================================
# ANÁLISE DE SEGMENTOS
# Recebe texto transcrito e retorna se é interessante para corte
# Critérios: emoção alta OU relevância temática alta OU risada detectada
# =============================================================================
def segmento_e_interessante(texto: str) -> bool:
    if len(texto.strip()) < 10:
        return False

    texto_lower = texto.lower()

    # Detecção de risada por marcadores textuais do Whisper
    # Limitação conhecida: "kkk" pode gerar falsos positivos com siglas
    marcadores_risada = ["[risos]", "(risos)", "haha", "kkk", "rsrs", "hehe"]
    tem_risada = any(m in texto_lower for m in marcadores_risada)

    if tem_risada:
        return True

    resultado_sentimento = modelo_sentimento(texto)[0]
    if resultado_sentimento["score"] > MIN_SCORE_SENTIMENTO:
        return True

    resultado_relevancia = modelo_relevancia(texto, candidate_labels=TEMAS_RELEVANTES)
    nota_relevancia = max(resultado_relevancia["scores"])
    if nota_relevancia > MIN_SCORE_RELEVANCIA:
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
            # Mescla: estende o trecho atual até o fim do próximo
            atual["end"] = max(atual["end"], proximo["end"])
        else:
            if atual["end"] - atual["start"] >= MIN_DURACAO:
                resultado.append(atual)
            atual = proximo.copy()

    # Não esquecer o último trecho
    if atual["end"] - atual["start"] >= MIN_DURACAO:
        resultado.append(atual)

    return resultado


# =============================================================================
# PIPELINE PRINCIPAL
# Fluxo: URL → áudio → transcrição → análise → mesclagem → cortes
# =============================================================================
def processar_url(url: str):
    print("=" * 50)
    print("INICIANDO PIPELINE DE CORTES AUTÔNOMOS")
    print("=" * 50)

    # 1. Extrai apenas o áudio para transcrição
    caminho_audio = extrair_audio(url)

    # 2. Transcrição com Whisper
    print("Transcrevendo áudio com Whisper...")
    resultado = modelo_whisper.transcribe(caminho_audio, fp16=(device_gpu == "cuda"))

    duracao_total = resultado["segments"][-1]["end"] if resultado["segments"] else 9999

    # 3. Identifica segmentos interessantes
    print("Analisando segmentos por sentimento e relevância...")
    pontos_interesse = []

    for seg in resultado["segments"]:
        texto = seg["text"].strip()

        if segmento_e_interessante(texto):
            pontos_interesse.append({
                "start": max(0, seg["start"] - BUFFER_TEMPO),
                "end": min(duracao_total, seg["end"] + BUFFER_TEMPO),
            })

    # 4. Mescla segmentos próximos
    cortes_finais = mesclar_segmentos(pontos_interesse)

    # 5. Gera os cortes em vídeo
    print(f"\nEncontrados {len(cortes_finais)} cortes. Iniciando downloads...\n")

    for i, trecho in enumerate(cortes_finais, 1):
        # Margem extra para garantir keyframe limpo no início/fim
        inicio_seguro = max(0, trecho["start"] - MARGEM_KEYFRAME)
        fim_seguro = min(duracao_total, trecho["end"] + MARGEM_KEYFRAME)
        nome_saida = f"corte_{i:02d}.mp4"

        cortar_video(url, inicio_seguro, fim_seguro, nome_saida)

    # 6. Limpeza
    if os.path.exists(caminho_audio):
        os.remove(caminho_audio)

    print("\nOperação concluída. Arquivos salvos em:", PASTA_SAIDA)


# =============================================================================
# ENTRADA
# =============================================================================
if __name__ == "__main__":
    url_alvo = input("URL do vídeo ou live (YouTube): ").strip()

    if url_alvo:
        processar_url(url_alvo)
    else:
        print("URL inválida. Encerrando.")
