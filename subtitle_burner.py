import os
import subprocess
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# =============================================================================
# SUBTITLE BURNER — Legendas estilo short-form (Opus Clip / Submagic)
#
# Gera arquivo .ass (Advanced SubStation Alpha) com 2-3 palavras por vez,
# em CAIXA ALTA, fonte grossa com contorno preto. Depois usa ffmpeg para
# "queimar" a legenda no vídeo (burn-in), tornando-a parte da imagem.
#
# Por que .ass e não .srt?
#   .srt é limitado a texto plano. .ass suporta estilos avançados (fonte,
#   cor, contorno, sombra, posição) que são essenciais para o visual de
#   short-form.
#
# Como funciona:
#   1. Recebe os "words" do Whisper (palavras com timestamps individuais)
#   2. Agrupa em chunks de 2-3 palavras
#   3. Gera o .ass com timing relativo ao corte (não ao vídeo original)
#   4. Chama ffmpeg com filtro subtitles para queimar
# =============================================================================

# Configurações visuais — ajuste aqui se quiser mudar o estilo
FONTE             = "Montserrat"
TAMANHO_FONTE     = 18           # escala automaticamente para 1080x1920
COR_TEXTO         = "&H00FFFFFF" # branco (formato ASS: AABBGGRR)
COR_CONTORNO      = "&H00000000" # preto
ESPESSURA_CONTORNO = 3
SOMBRA            = 1
POSICAO_VERTICAL  = 75           # % da altura da tela (75 = 3/4 para baixo)
PALAVRAS_POR_CHUNK = 3           # 2 ou 3 funciona bem; mais que isso polui


def _formatar_tempo_ass(segundos: float) -> str:
    """
    Converte segundos para o formato de tempo do ASS: H:MM:SS.cc
    (centésimos, não milissegundos).
    """
    horas      = int(segundos // 3600)
    minutos    = int((segundos % 3600) // 60)
    segs       = int(segundos % 60)
    centesimos = int((segundos - int(segundos)) * 100)
    return f"{horas}:{minutos:02d}:{segs:02d}.{centesimos:02d}"


def _agrupar_palavras(palavras: list, palavras_por_chunk: int) -> list:
    """
    Agrupa palavras em chunks de N palavras.
    Cada chunk tem início (primeira palavra) e fim (última palavra).
    """
    chunks = []
    for i in range(0, len(palavras), palavras_por_chunk):
        grupo = palavras[i:i + palavras_por_chunk]
        if not grupo:
            continue

        texto = " ".join(p["word"].strip() for p in grupo).upper()  # CAIXA ALTA
        chunks.append({
            "inicio": grupo[0]["start"],
            "fim":    grupo[-1]["end"],
            "texto":  texto,
        })
    return chunks


def gerar_ass(
    palavras: list,
    caminho_saida: str,
    inicio_corte: float,
    fim_corte: float,
) -> bool:
    """
    Gera arquivo .ass com legendas word-by-word para um corte específico.

    Args:
        palavras:      Lista de dicts {word, start, end} do Whisper word_timestamps
        caminho_saida: Caminho onde salvar o .ass
        inicio_corte:  Tempo de início do corte no vídeo original (segundos)
        fim_corte:     Tempo de fim do corte no vídeo original (segundos)

    Returns:
        True se gerou com sucesso, False caso contrário
    """
    # Filtra só as palavras que pertencem a este corte
    # E ajusta os timestamps para serem RELATIVOS ao início do corte
    palavras_do_corte = []
    for p in palavras:
        if p["start"] >= inicio_corte and p["end"] <= fim_corte:
            palavras_do_corte.append({
                "word":  p["word"],
                "start": p["start"] - inicio_corte,  # relativo ao corte
                "end":   p["end"]   - inicio_corte,
            })

    if not palavras_do_corte:
        log.warning(f"  Nenhuma palavra encontrada para o trecho {inicio_corte}-{fim_corte}s")
        return False

    chunks = _agrupar_palavras(palavras_do_corte, PALAVRAS_POR_CHUNK)

    # Cabeçalho do arquivo .ass
    # PlayResX/Y devem bater com a resolução do vídeo final (1080x1920)
    cabecalho = f"""[Script Info]
Title: Nexus subtitle
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONTE},{TAMANHO_FONTE * 6},{COR_TEXTO},{COR_TEXTO},{COR_CONTORNO},&H80000000,1,0,0,0,100,100,0,0,1,{ESPESSURA_CONTORNO},{SOMBRA},2,60,60,{int(1920 * (100 - POSICAO_VERTICAL) / 100)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Cada chunk vira uma linha "Dialogue" no .ass
    linhas_dialogo = []
    for chunk in chunks:
        inicio_str = _formatar_tempo_ass(chunk["inicio"])
        fim_str    = _formatar_tempo_ass(chunk["fim"])
        # Escapa caracteres especiais do ASS
        texto = chunk["texto"].replace("{", "(").replace("}", ")")
        linhas_dialogo.append(
            f"Dialogue: 0,{inicio_str},{fim_str},Default,,0,0,0,,{texto}"
        )

    try:
        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(cabecalho)
            f.write("\n".join(linhas_dialogo))
        return True
    except Exception as e:
        log.error(f"  Erro ao gerar .ass: {e}")
        return False


def queimar_legenda(
    video_entrada: str,
    arquivo_ass: str,
    video_saida: str,
) -> bool:
    """
    Usa ffmpeg para "queimar" a legenda do .ass no vídeo (burn-in).

    O filtro subtitles do ffmpeg precisa do path do .ass escapado de forma
    específica no Windows (caminho com : vira \\: para o filtro entender).
    """
    # Escapa o path para o filtro subtitles do ffmpeg
    # Necessário no Windows por causa do C:\ no caminho
    ass_path_escapado = arquivo_ass.replace("\\", "/").replace(":", "\\:")

    comando = [
        "ffmpeg", "-y",
        "-i", video_entrada,
        "-vf", f"subtitles='{ass_path_escapado}'",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "copy",  # áudio não muda, só copia
        video_saida,
    ]

    log.info(f"  Queimando legenda em {Path(video_saida).name}...")
    resultado = subprocess.run(comando, capture_output=True, text=True)

    if resultado.returncode != 0:
        log.error(f"  ffmpeg falhou ao queimar legenda:\n{resultado.stderr[-400:]}")
        return False

    return True


def aplicar_legenda(
    video_entrada: str,
    palavras: list,
    inicio_corte: float,
    fim_corte: float,
    video_saida: str,
) -> bool:
    """
    Função principal — gera o .ass, queima no vídeo e limpa o arquivo temporário.

    Args:
        video_entrada: Vídeo já cortado (sem legenda)
        palavras:      Word timestamps do Whisper para o vídeo INTEIRO
        inicio_corte:  Tempo de início do corte no vídeo original
        fim_corte:     Tempo de fim do corte no vídeo original
        video_saida:   Vídeo final com legenda queimada

    Returns:
        True se tudo deu certo
    """
    # Gera .ass temporário
    arquivo_ass = video_entrada.replace(".mp4", ".ass")

    if not gerar_ass(palavras, arquivo_ass, inicio_corte, fim_corte):
        return False

    # Queima no vídeo
    sucesso = queimar_legenda(video_entrada, arquivo_ass, video_saida)

    # Limpa o .ass temporário
    if os.path.exists(arquivo_ass):
        os.remove(arquivo_ass)

    return sucesso
