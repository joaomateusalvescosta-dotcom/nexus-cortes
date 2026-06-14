import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# =============================================================================
# RESULTADO DE PUBLICAÇÃO
# Dataclass retornada por todo publisher após tentativa de upload.
# Padroniza o que o publish.py recebe independente da plataforma.
# =============================================================================
@dataclass
class ResultadoPublicacao:
    plataforma: str
    arquivo: str
    sucesso: bool
    url_publicada: str = ""
    id_publicacao: str = ""
    erro: str = ""
    metadados_extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.sucesso:
            return f"[{self.plataforma}] ✓ {self.arquivo} → {self.url_publicada}"
        return f"[{self.plataforma}] ✗ {self.arquivo} — {self.erro}"


# =============================================================================
# CONTRATO BASE
# Todo publisher deve implementar:
#   - publicar(): envia o vídeo para a plataforma
#   - validar_credenciais(): verifica se as credenciais estão corretas antes de tentar
#
# Método utilitário já implementado aqui:
#   - _verificar_arquivo(): valida existência e tamanho do arquivo antes de enviar
# =============================================================================
class PublisherBase(ABC):

    NOME_PLATAFORMA: str = "desconhecida"

    # Limite de tamanho padrão (pode ser sobrescrito por subclasse)
    LIMITE_TAMANHO_MB: float = 500.0

    def __init__(self):
        self.log = logging.getLogger(f"publisher.{self.NOME_PLATAFORMA}")

    @abstractmethod
    def publicar(
        self,
        caminho_video: str,
        titulo: str,
        descricao: str = "",
        tags: list[str] | None = None,
    ) -> ResultadoPublicacao:
        """
        Publica o vídeo na plataforma.

        Args:
            caminho_video: Caminho absoluto ou relativo para o arquivo .mp4
            titulo:        Título do vídeo (cada plataforma tem limite próprio)
            descricao:     Descrição/legenda (opcional)
            tags:          Lista de hashtags sem # (opcional)

        Returns:
            ResultadoPublicacao com status, URL e metadados
        """
        ...

    @abstractmethod
    def validar_credenciais(self) -> bool:
        """
        Verifica se as credenciais da plataforma estão configuradas e válidas.
        Deve ser chamado antes de publicar para dar erro claro ao usuário.
        """
        ...

    def _verificar_arquivo(self, caminho_video: str) -> str | None:
        """
        Valida o arquivo de vídeo antes de tentar o upload.

        Returns:
            Mensagem de erro se inválido, None se tudo ok
        """
        path = Path(caminho_video)

        if not path.exists():
            return f"Arquivo não encontrado: {caminho_video}"

        if not path.suffix.lower() == ".mp4":
            return f"Formato inválido ({path.suffix}). Esperado: .mp4"

        tamanho_mb = path.stat().st_size / (1024 * 1024)
        if tamanho_mb > self.LIMITE_TAMANHO_MB:
            return (
                f"Arquivo muito grande: {tamanho_mb:.1f}MB "
                f"(limite da plataforma: {self.LIMITE_TAMANHO_MB}MB)"
            )

        return None
