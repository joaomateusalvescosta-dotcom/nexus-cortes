import logging
import os
import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import pickle

from .base import PublisherBase, ResultadoPublicacao

log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÃO
#
# Antes de usar:
# 1. Acesse https://console.cloud.google.com
# 2. Crie um projeto → ative "YouTube Data API v3"
# 3. Credentials → OAuth 2.0 → Desktop App → baixe o JSON
# 4. Renomeie para "client_secrets_youtube.json" e coloque na raiz do projeto
#
# O token é salvo em "token_youtube.pickle" após o primeiro login.
# Nas próximas execuções, o refresh é automático.
#
# Dependências:
#   pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
# =============================================================================

SCOPES                 = ["https://www.googleapis.com/auth/youtube.upload"]
ARQUIVO_SECRETS        = "client_secrets_youtube.json"
ARQUIVO_TOKEN          = "token_youtube.pickle"
LIMITE_TITULO          = 100    # limite real da API do YouTube
LIMITE_DESCRICAO       = 5000
LIMITE_TAGS            = 500    # total de caracteres somados em todas as tags


class YouTubePublisher(PublisherBase):

    NOME_PLATAFORMA   = "youtube"
    LIMITE_TAMANHO_MB = 128_000.0   # YouTube aceita até 128GB

    def __init__(self):
        super().__init__()
        self._servico = None

    # ------------------------------------------------------------------
    # AUTENTICAÇÃO
    # Fluxo OAuth 2.0 com cache em pickle.
    # Na primeira vez abre o browser para login; depois usa refresh token.
    # ------------------------------------------------------------------
    def _autenticar(self) -> bool:
        credenciais = None

        if os.path.exists(ARQUIVO_TOKEN):
            with open(ARQUIVO_TOKEN, "rb") as f:
                credenciais = pickle.load(f)

        if credenciais and credenciais.expired and credenciais.refresh_token:
            self.log.info("Renovando token do YouTube...")
            credenciais.refresh(Request())

        elif not credenciais or not credenciais.valid:
            if not os.path.exists(ARQUIVO_SECRETS):
                self.log.error(
                    f"Arquivo '{ARQUIVO_SECRETS}' não encontrado. "
                    "Baixe em: console.cloud.google.com → Credentials → OAuth 2.0"
                )
                return False

            flow = InstalledAppFlow.from_client_secrets_file(ARQUIVO_SECRETS, SCOPES)
            credenciais = flow.run_local_server(port=0)

        with open(ARQUIVO_TOKEN, "wb") as f:
            pickle.dump(credenciais, f)

        self._servico = build("youtube", "v3", credentials=credenciais)
        return True

    def validar_credenciais(self) -> bool:
        if self._servico:
            return True
        return self._autenticar()

    # ------------------------------------------------------------------
    # PUBLICAÇÃO
    # Usa upload resumable (MediaFileUpload) para suportar arquivos grandes.
    # Retry automático em erros 5xx (problemas transitórios da API).
    # ------------------------------------------------------------------
    def publicar(
        self,
        caminho_video: str,
        titulo: str,
        descricao: str = "",
        tags: list[str] | None = None,
        privacidade: str = "public",
    ) -> ResultadoPublicacao:

        # Validações locais antes de tocar na API
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
                erro="Falha na autenticação. Verifique client_secrets_youtube.json."
            )

        titulo_truncado    = titulo[:LIMITE_TITULO]
        descricao_truncada = descricao[:LIMITE_DESCRICAO]

        # Trunca tags para não ultrapassar o limite de caracteres total
        tags_validas = []
        total_chars  = 0
        for tag in (tags or []):
            tag_limpa = tag.strip().lstrip("#")
            if total_chars + len(tag_limpa) <= LIMITE_TAGS:
                tags_validas.append(tag_limpa)
                total_chars += len(tag_limpa)

        corpo = {
            "snippet": {
                "title":       titulo_truncado,
                "description": descricao_truncada,
                "tags":        tags_validas,
                "categoryId":  "22",    # 22 = People & Blogs (mais genérico para short-form)
            },
            "status": {
                "privacyStatus": privacidade,
                # Marca como Short via título — a API não tem campo dedicado.
                # O YouTube detecta automaticamente pelo formato vertical 9:16.
            }
        }

        media = MediaFileUpload(
            caminho_video,
            mimetype="video/mp4",
            resumable=True,     # upload em chunks — tolerante a instabilidade de rede
            chunksize=10 * 1024 * 1024  # 10MB por chunk
        )

        self.log.info(f"Iniciando upload para YouTube: {Path(caminho_video).name}")

        MAX_TENTATIVAS = 3
        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                requisicao = self._servico.videos().insert(
                    part="snippet,status",
                    body=corpo,
                    media_body=media
                )

                resposta  = None
                progresso_anterior = -1

                # Loop de upload em chunks com log de progresso
                while resposta is None:
                    status, resposta = requisicao.next_chunk()
                    if status:
                        progresso = int(status.progress() * 100)
                        if progresso != progresso_anterior:
                            self.log.info(f"  Upload YouTube: {progresso}%")
                            progresso_anterior = progresso

                video_id  = resposta["id"]
                url_video = f"https://www.youtube.com/shorts/{video_id}"

                self.log.info(f"  ✓ Publicado: {url_video}")

                return ResultadoPublicacao(
                    plataforma    = self.NOME_PLATAFORMA,
                    arquivo       = caminho_video,
                    sucesso       = True,
                    url_publicada = url_video,
                    id_publicacao = video_id,
                    metadados_extras = {"titulo": titulo_truncado, "tags": tags_validas}
                )

            except HttpError as e:
                codigo = e.resp.status
                if codigo in (500, 502, 503, 504) and tentativa < MAX_TENTATIVAS:
                    espera = 10 * tentativa
                    self.log.warning(f"  Erro {codigo} na tentativa {tentativa}. Aguardando {espera}s...")
                    time.sleep(espera)
                else:
                    return ResultadoPublicacao(
                        plataforma=self.NOME_PLATAFORMA,
                        arquivo=caminho_video,
                        sucesso=False,
                        erro=f"HttpError {codigo}: {e.reason}"
                    )

            except Exception as e:
                return ResultadoPublicacao(
                    plataforma=self.NOME_PLATAFORMA,
                    arquivo=caminho_video,
                    sucesso=False,
                    erro=str(e)
                )

        return ResultadoPublicacao(
            plataforma=self.NOME_PLATAFORMA,
            arquivo=caminho_video,
            sucesso=False,
            erro=f"Falha após {MAX_TENTATIVAS} tentativas."
        )
