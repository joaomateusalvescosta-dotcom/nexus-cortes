import logging
import os
import glob
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from publisher import YouTubePublisher, ResultadoPublicacao
from metadata_generator import gerar_metadados

from dotenv import load_dotenv
load_dotenv()
# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
PASTA_CORTES      = "cortes_finalizados"
PASTA_TRANSCRIPTS = "transcripts"   # gerado pela main_nexus1 (ver abaixo)

# -----------------------------------------------------------------------------
# CONTROLE DE QUOTA DO YOUTUBE
#
# A API do YouTube tem cota diária de 10.000 unidades no plano gratuito.
# Cada upload custa ~1.600 unidades → máximo de ~6 uploads por dia.
# Para não estourar a cota, postamos apenas os N cortes de MAIOR score.
#
# O score está no próprio nome do arquivo (corte_01_score84.mp4 → 84),
# então ordenamos por ele e pegamos os melhores.
# -----------------------------------------------------------------------------
MAX_UPLOADS_POR_EXECUCAO = 5   # fica abaixo do limite de 6 por segurança

# -----------------------------------------------------------------------------
# PRIVACIDADE DOS VÍDEOS
#   "private"  → só você vê (ideal para TESTAR)
#   "unlisted" → quem tem o link vê, não aparece no canal
#   "public"   → todo mundo vê (produção real)
#
# Para o primeiro teste, deixe "private" para não despejar vídeos no canal.
# -----------------------------------------------------------------------------
PRIVACIDADE_TESTE = "private"

# Plataformas ativas
PUBLISHERS_ATIVOS = [
    YouTubePublisher(),
    # InstagramPublisher(),  # futuro
]


# =============================================================================
# EXTRAI O SCORE DO NOME DO ARQUIVO
# corte_01_score84.mp4 → 84
# Usado para ordenar os cortes e publicar só os melhores.
# =============================================================================
def extrair_score(caminho: str) -> int:
    nome = Path(caminho).stem
    try:
        # Procura "score" no nome e pega o número depois
        if "score" in nome:
            return int(nome.split("score")[-1])
    except (ValueError, IndexError):
        pass
    return 0


# =============================================================================
# CARREGA METADADOS SALVOS PELA MAIN_NEXUS1
#
# A main_nexus1 salva um JSON por corte em transcripts/ com:
#   {
#     "transcript": "...",
#     "temas":      ["conselho", ...],
#     "score":      78
#   }
#
# Isso evita re-transcrever o vídeo na hora de publicar.
# Se o JSON não existir, publica com metadados mínimos de fallback.
# =============================================================================
def carregar_contexto_corte(nome_mp4: str) -> dict:
    nome_base = Path(nome_mp4).stem
    caminho_json = os.path.join(PASTA_TRANSCRIPTS, f"{nome_base}.json")

    if os.path.exists(caminho_json):
        try:
            with open(caminho_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Erro ao ler {caminho_json}: {e}")

    # Fallback mínimo se o JSON não existir
    return {"transcript": "", "temas": [], "score": 0}


# =============================================================================
# PUBLICAÇÃO EM PARALELO POR PLATAFORMA
# =============================================================================
def publicar_arquivo(
    caminho: str,
    publishers: list,
    titulo: str,
    descricao: str,
    tags: list[str],
) -> list[ResultadoPublicacao]:

    resultados = []

    def enviar(publisher):
        log.info(f"[{publisher.NOME_PLATAFORMA}] Publicando {Path(caminho).name}...")
        # YouTube aceita privacidade; outros publishers ignoram via try
        try:
            return publisher.publicar(caminho, titulo, descricao, tags, privacidade=PRIVACIDADE_TESTE)
        except TypeError:
            # Publisher que não aceita o parâmetro privacidade (ex: TikTok)
            return publisher.publicar(caminho, titulo, descricao, tags)

    with ThreadPoolExecutor(max_workers=len(publishers)) as executor:
        futuros = {executor.submit(enviar, p): p for p in publishers}
        for futuro in as_completed(futuros):
            try:
                resultado = futuro.result()
                resultados.append(resultado)
                log.info(str(resultado))
            except Exception as e:
                plataforma = futuros[futuro].NOME_PLATAFORMA
                log.error(f"[{plataforma}] Erro inesperado: {e}")

    return resultados


# =============================================================================
# REVISÃO INTERATIVA
#
# Exibe os metadados gerados e permite editar antes de confirmar.
# Fluxo por corte:
#   1. Mostra título, descrição e hashtags gerados pela IA
#   2. Usuário pode confirmar (Enter), editar campo a campo, ou pular o corte
#
# Esse passo é opcional — para rodar sem interação (modo automático full),
# defina a variável de ambiente NEXUS_AUTO=1
# =============================================================================
MODO_AUTO = os.getenv("NEXUS_AUTO", "0") == "1"


def revisar_metadados(metadados: dict, nome_arquivo: str) -> dict | None:
    """
    Apresenta os metadados gerados e permite edição interativa.
    Retorna os metadados revisados, ou None se o usuário pular o corte.
    """
    if MODO_AUTO:
        return metadados

    print("\n" + "─" * 55)
    print(f"  REVISÃO: {nome_arquivo}")
    print("─" * 55)
    print(f"  Título    : {metadados['titulo']}")
    print(f"  Descrição : {metadados['descricao']}")
    print(f"  Hashtags  : {' '.join('#' + h for h in metadados['hashtags'])}")
    print("─" * 55)
    print("  [Enter] Confirmar  [e] Editar  [s] Pular este corte")

    escolha = input("  > ").strip().lower()

    if escolha == "s":
        log.info(f"Corte pulado: {nome_arquivo}")
        return None

    if escolha == "e":
        print("\n  Edite os campos (Enter para manter o valor atual):\n")

        novo_titulo = input(f"  Título [{metadados['titulo']}]: ").strip()
        if novo_titulo:
            metadados["titulo"] = novo_titulo[:80]

        nova_desc = input(f"  Descrição [{metadados['descricao']}]: ").strip()
        if nova_desc:
            metadados["descricao"] = nova_desc[:200]

        print(f"  Hashtags atuais: {' '.join('#' + h for h in metadados['hashtags'])}")
        novas_tags = input("  Novas hashtags (separadas por espaço, sem #): ").strip()
        if novas_tags:
            metadados["hashtags"] = [t.lstrip("#") for t in novas_tags.split()]

        print("\n  Metadados finais:")
        print(f"  Título    : {metadados['titulo']}")
        print(f"  Descrição : {metadados['descricao']}")
        print(f"  Hashtags  : {' '.join('#' + h for h in metadados['hashtags'])}")
        confirmar = input("\n  Confirmar? [Enter = sim / n = pular]: ").strip().lower()
        if confirmar == "n":
            return None

    return metadados


# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
def exibir_relatorio(todos_resultados: list[ResultadoPublicacao]):
    log.info("\n" + "=" * 55)
    log.info("RELATÓRIO DE PUBLICAÇÃO")
    log.info("=" * 55)

    por_plataforma: dict = {}
    for r in todos_resultados:
        por_plataforma.setdefault(r.plataforma, []).append(r)

    for plataforma, resultados in por_plataforma.items():
        sucessos = [r for r in resultados if r.sucesso]
        falhas   = [r for r in resultados if not r.sucesso]
        log.info(f"\n{plataforma.upper()}: {len(sucessos)} publicados, {len(falhas)} falhas")
        for r in sucessos:
            log.info(f"  ✓ {Path(r.arquivo).name} → {r.url_publicada}")
        for r in falhas:
            log.info(f"  ✗ {Path(r.arquivo).name} — {r.erro}")


# =============================================================================
# VALIDAÇÃO DE CREDENCIAIS
# =============================================================================
def validar_todos_publishers(publishers: list) -> list:
    validos = []
    for p in publishers:
        if p.validar_credenciais():
            log.info(f"[{p.NOME_PLATAFORMA}] Credenciais OK")
            validos.append(p)
        else:
            log.warning(f"[{p.NOME_PLATAFORMA}] Credenciais inválidas — plataforma desativada")
    return validos


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("=" * 55)
    log.info("NEXUS — MÓDULO DE PUBLICAÇÃO")
    if MODO_AUTO:
        log.info("Modo: AUTOMÁTICO (NEXUS_AUTO=1)")
    else:
        log.info("Modo: REVISÃO INTERATIVA")
    log.info(f"Privacidade: {PRIVACIDADE_TESTE.upper()}")
    log.info("=" * 55)

    publishers_validos = validar_todos_publishers(PUBLISHERS_ATIVOS)
    if not publishers_validos:
        log.error("Nenhuma plataforma com credenciais válidas. Encerrando.")
        return

    arquivos = sorted(glob.glob(os.path.join(PASTA_CORTES, "corte_*.mp4")))
    if not arquivos:
        log.warning(f"Nenhum corte encontrado em '{PASTA_CORTES}/'. Execute main_nexus1.py primeiro.")
        return

    # Ordena por score (maior primeiro) e pega só os N melhores
    # para não estourar a cota diária do YouTube
    arquivos_ordenados = sorted(arquivos, key=extrair_score, reverse=True)
    arquivos_publicar  = arquivos_ordenados[:MAX_UPLOADS_POR_EXECUCAO]

    log.info(f"\n{len(arquivos)} corte(s) encontrado(s).")
    log.info(f"Publicando os {len(arquivos_publicar)} de maior score (limite de cota):\n")
    for caminho in arquivos_publicar:
        log.info(f"  • {Path(caminho).name} (score {extrair_score(caminho)})")
    log.info("")

    todos_resultados: list[ResultadoPublicacao] = []

    for caminho in arquivos_publicar:
        nome = Path(caminho).name

        # 1. Carrega transcript e contexto gerado pela main_nexus1
        contexto = carregar_contexto_corte(nome)

        # 2. Gera metadados via IA (ou fallback por regras)
        metadados = gerar_metadados(
            transcript       = contexto.get("transcript", ""),
            temas_detectados = contexto.get("temas", []),
            score            = contexto.get("score", 0),
            nome_arquivo     = nome,
        )

        # 3. Revisão interativa (ou passa direto no modo auto)
        metadados_finais = revisar_metadados(metadados, nome)
        if metadados_finais is None:
            continue

        # 4. Publica
        resultados = publicar_arquivo(
            caminho    = caminho,
            publishers = publishers_validos,
            titulo     = metadados_finais["titulo"],
            descricao  = metadados_finais["descricao"],
            tags       = metadados_finais["hashtags"],
        )
        todos_resultados.extend(resultados)

    exibir_relatorio(todos_resultados)


if __name__ == "__main__":
    main()
