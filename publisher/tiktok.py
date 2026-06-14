import logging
import os
import time
import requests
from pathlib import Path

from .base import PublisherBase, ResultadoPublicacao

log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÃO
#
# A TikTok Content Posting API exige aprovação prévia do seu app:
# 1. Acesse https://developers.tiktok.com
# 2. Crie um app → solicite o produto "Content Posting API"
#   (aprovação pode levar alguns dias — justifique como projeto de automação)
# 3. Após aprovação, gere um Access Token via OAuth 2.0
# 4. Defina as variáveis de ambiente:
#      TIKTOK_ACCESS_TOKEN=...
#      TIKTOK_CLIENT_KEY=...
#      TIKTOK_CLIENT_SECRET=...
#
# Documentação: https://developers.tiktok.com/doc/content-posting-api-get-started
#
# Dependências:
#   pip install requests
#
# LIMITAÇÃO CONHECIDA:
#   O TikTok não fornece URL pública do vídeo na resposta do upload.
#   O publish_id retornado pode ser usado para consultar o status via
#   POST /v2/post/publish/status/fetch/ (implementado em _consultar_status).
# =============================================================================

TIKTOK_API_BASE      = "https://open.tiktokapis.com/v2"
LIMITE_TITULO_TIKTOK = 150   # TikTok chama de "caption" — inclui hashtags
LIMITE_TAMANHO_MB    = 4_000.0   # TikTok aceita até 4GB via upload direto


class TikTokPublisher(PublisherBase):

    NOME_PLATAFORMA   = "tiktok"
    LIMITE_TAMANHO_MB = LIMITE_TAMANHO_MB

    def __init__(self):
        super().__init__()
        self._access_token   = os.getenv("TIKTOK_ACCESS_TOKEN", "")
        self._client_key     = os.getenv("TIKTOK_CLIENT_KEY", "")
        self._client_secret  = os.getenv("TIKTOK_CLIENT_SECRET", "")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type":  "application/json; charset=UTF-8",
        }

    # ------------------------------------------------------------------
    # VALIDAÇÃO DE CREDENCIAIS
    # Tenta buscar as informações do creator — se retornar 200, o token é válido.
    # ------------------------------------------------------------------
    def validar_credenciais(self) -> bool:
        if not self._access_token:
            self.log.error(
                "TIKTOK_ACCESS_TOKEN não definido. "
                "Configure a variável de ambiente antes de publicar."
            )
            return False

        try:
            resp = requests.post(
                f"{TIKTOK_API_BASE}/post/publish/creator_info/query/",
                headers=self._headers(),
                json={},
                timeout=10
            )
            if resp.status_code == 200:
                return True

            self.log.error(f"Token inválido ou expirado. Status: {resp.status_code} — {resp.text}")
            return False

        except requests.RequestException as e:
            self.log.error(f"Erro ao validar credenciais TikTok: {e}")
            return False

    # ------------------------------------------------------------------
    # CONSULTA DE STATUS
    # Após o upload, o TikTok processa o vídeo de forma assíncrona.
    # Este método policia o status até o vídeo estar publicado ou falhar.
    # ------------------------------------------------------------------
    def _consultar_status(self, publish_id: str, max_tentativas: int = 15) -> dict:
        for i in range(max_tentativas):
            try:
                resp = requests.post(
                    f"{TIKTOK_API_BASE}/post/publish/status/fetch/",
                    headers=self._headers(),
                    json={"publish_id": publish_id},
                    timeout=10
                )
                dados = resp.json().get("data", {})
                status = dados.get("status", "UNKNOWN")

                self.log.info(f"  Status TikTok [{i+1}/{max_tentativas}]: {status}")

                if status == "PUBLISH_COMPLETE":
                    return {"sucesso": True, "status": status}

                if status in ("FAILED", "CANCELLED"):
                    motivo = dados.get("fail_reason", "desconhecido")
                    return {"sucesso": False, "status": status, "motivo": motivo}

                time.sleep(8)   # TikTok recomenda polling a cada 5–10s

            except Exception as e:
                self.log.warning(f"  Erro ao consultar status: {e}")
                time.sleep(5)

        return {"sucesso": False, "status": "TIMEOUT", "motivo": "Timeout ao aguardar publicação"}

    # ------------------------------------------------------------------
    # PUBLICAÇÃO
    # Fluxo de 3 etapas da Content Posting API:
    #   1. Inicializar upload → receber upload_url e publish_id
    #   2. Enviar o arquivo de vídeo para upload_url (PUT direto)
    #   3. Consultar status até publicação concluída
    # ------------------------------------------------------------------
    def publicar(
        self,
        caminho_video: str,
        titulo: str,
        descricao: str = "",
        tags: list[str] | None = None,
    ) -> ResultadoPublicacao:

        erro_arquivo = self._verificar_arquivo(caminho_video)
        if erro_arquivo:
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro=erro_arquivo
            )

        if not self.validar_credenciais():
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro="Falha na autenticação. Verifique TIKTOK_ACCESS_TOKEN."
            )

        # Monta a caption: título + hashtags (tudo no mesmo campo no TikTok)
        hashtags = " ".join(f"#{t.strip().lstrip('#')}" for t in (tags or []))
        caption  = f"{titulo} {hashtags}".strip()[:LIMITE_TITULO_TIKTOK]

        tamanho_bytes = Path(caminho_video).stat().st_size

        # ------------------------------------------------------------------
        # ETAPA 1 — Inicializa o upload
        # ------------------------------------------------------------------
        self.log.info("Inicializando upload TikTok...")
        try:
            resp_init = requests.post(
                f"{TIKTOK_API_BASE}/post/publish/video/init/",
                headers=self._headers(),
                json={
                    "post_info": {
                        "title":          caption,
                        "privacy_level":  "PUBLIC_TO_EVERYONE",
                        "disable_duet":   False,
                        "disable_stitch": False,
                        "disable_comment": False,
                    },
                    "source_info": {
                        "source":          "FILE_UPLOAD",
                        "video_size":      tamanho_bytes,
                        "chunk_size":      tamanho_bytes,   # upload em chunk único
                        "total_chunk_count": 1,
                    }
                },
                timeout=15
            )
        except requests.RequestException as e:
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro=f"Falha ao inicializar upload: {e}"
            )

        if resp_init.status_code != 200:
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro=f"Erro na inicialização: {resp_init.status_code} — {resp_init.text}"
            )

        dados_init = resp_init.json().get("data", {})
        upload_url = dados_init.get("upload_url", "")
        publish_id = dados_init.get("publish_id", "")

        if not upload_url or not publish_id:
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro=f"Resposta inesperada da API: {resp_init.text}"
            )

        # ------------------------------------------------------------------
        # ETAPA 2 — Envia o vídeo
        # ------------------------------------------------------------------
        self.log.info(f"Enviando vídeo para TikTok ({tamanho_bytes / 1024 / 1024:.1f}MB)...")
        try:
            with open(caminho_video, "rb") as f:
                dados_video = f.read()

            resp_upload = requests.put(
                upload_url,
                data=dados_video,
                headers={
                    "Content-Type":  "video/mp4",
                    "Content-Range": f"bytes 0-{tamanho_bytes - 1}/{tamanho_bytes}",
                },
                timeout=300   # vídeos podem ser grandes — timeout generoso
            )
        except requests.RequestException as e:
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro=f"Falha no envio do vídeo: {e}"
            )

        if resp_upload.status_code not in (200, 201, 206):
            return ResultadoPublicacao(
                plataforma=self.NOME_PLATAFORMA,
                arquivo=caminho_video,
                sucesso=False,
                erro=f"Erro no upload: {resp_upload.status_code}"
            )

        self.log.info("Vídeo enviado. Aguardando processamento...")

        # ------------------------------------------------------------------
        # ETAPA 3 — Aguarda publicação
        # ------------------------------------------------------------------
        resultado_status = self._consultar_status(publish_id)

        if resultado_status["sucesso"]:
            # TikTok não retorna URL direta — o link padrão do perfil é o mais próximo
            self.log.info(f"  ✓ Publicado no TikTok (publish_id: {publish_id})")
            return ResultadoPublicacao(
                plataforma    = self.NOME_PLATAFORMA,
                arquivo       = caminho_video,
                sucesso       = True,
                url_publicada = "https://www.tiktok.com",   # URL genérica — API não retorna link direto
                id_publicacao = publish_id,
                metadados_extras = {"caption": caption, "status_final": resultado_status["status"]}
            )

        return ResultadoPublicacao(
            plataforma=self.NOME_PLATAFORMA,
            arquivo=caminho_video,
            sucesso=False,
            erro=f"Falha no processamento: {resultado_status.get('motivo', 'desconhecido')}"
        )
