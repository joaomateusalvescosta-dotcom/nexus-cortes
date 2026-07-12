# Nexus Cortes

Pipeline de estudo que automatiza a criação de cortes verticais para redes sociais. Recebe uma URL de um vídeo longo do YouTube (podcast, entrevista, live) e devolve cortes prontos para publicar, com legenda e metadados gerados por IA.

## O que faz

A partir de uma URL, o pipeline:

- baixa o áudio e transcreve com Whisper
- analisa cada trecho por sentimento e relevância para achar os melhores momentos
- pontua cada corte com um score de engajamento e descarta os fracos
- ajusta o fim do corte para não interromper falas no meio
- gera o vídeo vertical 9:16 com fundo borrado e legenda estilo Shorts/Reels
- cria título e hashtags com IA e publica no YouTube

## Organização

```
main_nexus2.py          Pipeline principal
subtitle_burner.py      Gera e queima as legendas
metadata_generator.py   Cria título, descrição e hashtags
publish.py              Publica (com revisão antes de postar)
publisher/
├── base.py             Classe base dos publishers
├── youtube.py          Publisher do YouTube
└── __init__.py
```

Os publishers ficam separados em módulos pra dar pra adicionar Instagram ou TikTok depois sem mexer no resto, é só criar um arquivo herdando da base.

## Decisões técnicas

**Modelos de análise na CPU.** Rodar Whisper + XLM-RoBERTa + DistilBART juntos estoura a VRAM numa GPU de 4GB. Deixei o Whisper na GPU e o resto na CPU.

**Download do vídeo completo antes de cortar.** Cortar direto pelo yt-dlp gerava arquivo sem stream válido em vários formatos. Baixar uma vez e cortar com ffmpeg localmente é mais estável, e ainda reaproveita o vídeo se rodar de novo.

**Fim do corte baseado em pausa real.** O Whisper erra pontuação demais em português, então o corte termina no silêncio real entre as falas em vez de confiar no ponto final. Os valores de pausa vêm de estudos de fala em português brasileiro.

## Limitações conhecidas

- A calibração dos scores é sensível e precisa de ajuste manual dependendo do tipo de conteúdo.
- Ainda saem cortes sem contexto — um trecho pode ser bom mas depender do que foi dito antes pra fazer sentido. Resolver isso precisa de análise semântica, não só detecção de pausa. É o principal ponto em aberto.

## Rodando

```bash
pip install -r requirements.txt
```

Precisa de **ffmpeg** e **Deno** no PATH, e a fonte **Montserrat** instalada.

Cria um `.env` na raiz:

```
GEMINI_API_KEY=sua_chave
NEXUS_AUTO=0
```

Pra publicar no YouTube, coloca o `client_secrets_youtube.json` na raiz (pega no Google Cloud Console).

Gerar os cortes:

```bash
python main_nexus2.py
```

Publicar:

```bash
python publish.py
```

## Stack

Whisper (transcrição), Transformers com XLM-RoBERTa e DistilBART (análise), yt-dlp (download), ffmpeg (edição e legenda), Gemini Flash (metadados), YouTube Data API v3 (publicação).

## Status

Funciona por inteiro pro YouTube. Instagram e TikTok ficam pra depois (TikTok foi removido por enquanto porque a API exige uma aprovação de app que não saiu).
