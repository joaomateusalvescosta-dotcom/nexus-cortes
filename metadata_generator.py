import json
import logging
import requests
import os

log = logging.getLogger(__name__)

# =============================================================================
# GERADOR DE METADADOS VIA IA
#
# Usa a API do Gemini Flash (Google) para gerar título, descrição e hashtags
# automaticamente a partir do transcript do corte e dos temas detectados.
#
# HASHTAGS PRÉ-SELECIONADAS POR NICHO:
#   Edite o dicionário HASHTAGS_POR_NICHO abaixo para customizar para o
#   seu canal. As hashtags do nicho detectado são somadas às geradas pela IA.
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

# ------------------------------------------------------------------
# HASHTAGS PRÉ-SELECIONADAS POR NICHO
# ------------------------------------------------------------------
HASHTAGS_POR_NICHO: dict[str, list[str]] = {
    "conselho": [
        "dicas", "aprendizado", "desenvolvimento", "crescimento",
        "autoconhecimento", "mentalidade"
    ],
    "polêmica": [
        "polemica", "debate", "opiniao", "verdade", "semlimite",
        "corte", "viral"
    ],
    "insight valioso": [
        "insight", "conhecimento", "reflexao", "sabedoria",
        "aprendizado", "mindset"
    ],
    "história curiosa": [
        "historia", "curiosidade", "fatos", "vocsabia",
        "historiacuriosa", "incrivel"
    ],
    "piada": [
        "humor", "risada", "engracado", "comedia", "memes",
        "funny", "haha"
    ],
    "futebol": [
        "futebol", "football", "gol", "brasileirao", "futebolbrasileiro",
        "golaca", "esporte", "campeonato"
    ],
    "humor": [
        "humor", "comedia", "risada", "engracado", "memes",
        "haha", "rir", "zueira" , "beta"
    ],
}

# Hashtags fixas que vão em todo corte independente do nicho
HASHTAGS_FIXAS = ["shorts", "cortes", "podcast", "viral"]

# Limite de hashtags no total
MAX_HASHTAGS_TOTAL = 8


# =============================================================================
# PROMPT PARA O GEMINI
# =============================================================================
PROMPT_SISTEMA = """Você é um especialista em criação de conteúdo para YouTube Shorts e redes sociais brasileiras.
Sua tarefa é gerar metadados otimizados para um corte de vídeo curto (short-form content).

Regras obrigatórias:
- Título: máximo 80 caracteres, deve ser chamativo e refletir o conteúdo real
- Título deve começar com um emoji relevante e criativo
- Descrição: máximo 200 caracteres, deve reforçar o gancho e incentivar o watch time
- Hashtags: retorne exatamente 5 hashtags sem #, relevantes ao conteúdo
- Responda APENAS com JSON válido, sem texto adicional, sem markdown, sem blocos de código

Formato de resposta:
{
  "titulo": "...",
  "descricao": "...",
  "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}"""


def _chamar_api(transcript: str, temas_detectados: list[str], score: int) -> dict | None:
    """
    Chama a API do Gemini Flash e retorna o JSON com título, descrição e hashtags.
    Retorna None se a chamada falhar.
    """
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY não definida — metadados gerados por fallback.")
        return None

    prompt_completo = f"""{PROMPT_SISTEMA}

Transcript do corte (fala original):
\"\"\"{transcript[:1500]}\"\"\"

Temas detectados pela análise: {", ".join(temas_detectados) if temas_detectados else "geral"}
Score de engajamento preditivo: {score}/100

Gere os metadados para este corte."""

    try:
        resposta = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt_completo}]}],
                "generationConfig": {
                    "temperature": 0.9,      # Alta criatividade para títulos mais chamativos
                    "maxOutputTokens": 300,
                    "responseMimeType": "application/json",  # Força resposta em JSON puro
                },
            },
            timeout=20,
        )

        if resposta.status_code != 200:
            log.warning(f"API Gemini retornou {resposta.status_code}: {resposta.text[:200]}")
            return None

        # Extrai o texto gerado pelo Gemini
        texto = (
            resposta.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        if not texto:
            log.warning("Gemini retornou resposta vazia.")
            return None

        # Remove possíveis blocos de markdown caso o modelo ignore a instrução
        texto = texto.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        return json.loads(texto)

    except json.JSONDecodeError as e:
        log.warning(f"Resposta da API não é JSON válido: {e}")
        return None
    except Exception as e:
        log.warning(f"Erro ao chamar API Gemini: {e}")
        return None


def _montar_hashtags(
    hashtags_ia: list[str],
    temas_detectados: list[str],
) -> list[str]:
    """
    Combina hashtags da IA + hashtags do nicho + hashtags fixas,
    remove duplicatas e respeita o limite total.
    """
    hashtags_nicho: list[str] = []
    for tema in temas_detectados:
        hashtags_nicho.extend(HASHTAGS_POR_NICHO.get(tema, []))

    # Ordem de prioridade: IA → nicho → fixas
    todas = hashtags_ia + hashtags_nicho + HASHTAGS_FIXAS

    vistas: set[str] = set()
    unicas: list[str] = []
    for tag in todas:
        tag_limpa = tag.strip().lower().lstrip("#")
        if tag_limpa and tag_limpa not in vistas:
            vistas.add(tag_limpa)
            unicas.append(tag_limpa)

    return unicas[:MAX_HASHTAGS_TOTAL]


def _fallback_metadados(
    transcript: str,
    temas_detectados: list[str],
    score: int,
    nome_arquivo: str,
) -> dict:
    """
    Geração de metadados por regras simples quando a API não está disponível.
    Funcional e sem dependências externas.
    """
    tema_principal = temas_detectados[0] if temas_detectados else "corte"

    emojis_por_tema = {
        "conselho":        "💡",
        "polêmica":        "🔥",
        "insight valioso": "🧠",
        "história curiosa": "😮",
        "piada":           "😂",
        "futebol":         "⚽",
        "humor":           "😂",
    }
    emoji = emojis_por_tema.get(tema_principal, "🎬")

    primeiras_palavras = " ".join(transcript.split()[:10]).strip()
    if primeiras_palavras:
        titulo = f"{emoji} \"{primeiras_palavras}...\""
    else:
        titulo = f"{emoji} Corte {score}/100 de engajamento"

    titulo = titulo[:80]
    descricao = f"Score de engajamento: {score}/100 • Tema: {tema_principal}"
    hashtags_nicho = HASHTAGS_POR_NICHO.get(tema_principal, [])[:3]

    return {
        "titulo":    titulo,
        "descricao": descricao,
        "hashtags":  _montar_hashtags(hashtags_nicho, temas_detectados),
    }


# =============================================================================
# FUNÇÃO PRINCIPAL — chamada pelo publish.py
# =============================================================================
def gerar_metadados(
    transcript: str,
    temas_detectados: list[str],
    score: int,
    nome_arquivo: str,
) -> dict:
    """
    Gera título, descrição e hashtags para um corte.

    Args:
        transcript:        Texto transcrito do trecho (fala original)
        temas_detectados:  Lista de temas identificados pela análise zero-shot
        score:             Score de engajamento preditivo (0–100)
        nome_arquivo:      Nome do arquivo .mp4 (usado no fallback)

    Returns:
        {
            "titulo":    str,
            "descricao": str,
            "hashtags":  list[str]
        }
    """
    log.info("  Gerando metadados via Gemini Flash...")

    resultado_ia = _chamar_api(transcript, temas_detectados, score)

    if resultado_ia:
        hashtags_finais = _montar_hashtags(
            resultado_ia.get("hashtags", []),
            temas_detectados,
        )
        metadados = {
            "titulo":    resultado_ia.get("titulo", "")[:80],
            "descricao": resultado_ia.get("descricao", "")[:200],
            "hashtags":  hashtags_finais,
        }
        log.info(f"  Título: {metadados['titulo']}")
        log.info(f"  Hashtags: {', '.join('#' + h for h in metadados['hashtags'])}")
        return metadados

    log.info("  Usando geração por regras (fallback)...")
    return _fallback_metadados(transcript, temas_detectados, score, nome_arquivo)
