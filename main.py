import os
import torch
import whisper
from moviepy import VideoFileClip
from transformers import pipeline

# --- ⚙️ CONSOLE DE CALIBRAÇÃO (LAPIDAÇÃO) ---
MIN_SCORE_SENT = 0.45      # Força da emoção
MIN_SCORE_PONTOS = 0.35    # Relevância do tema
MIN_DURATION = 55.0        # Duração mínima do corte (menos que isso vira lixo)
MAX_GAP_MERGE = 4.0       # Se a próxima fala boa vier em até 4s, JUNTA TUDO
BUFFER_TIME = 1.5         # "Gordurinha" de tempo antes e depois do corte

# --- 1. HARDWARE ---
device_gpu = "cuda" if torch.cuda.is_available() else "cpu"
device_cpu = "cpu"

# --- 2. CARREGAMENTO DAS IAs ---
print("🧠 Carregando os motores de decisão...")
modelo_ia = whisper.load_model("tiny", device=device_gpu)
sentimento_ai = pipeline("text-classification", model="pysentimiento/bertweet-pt-sentiment", device=-1)
pontuador_ai = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1)

# --- 3. CONFIGURAÇÕES ---
temas_relevantes = ["conselho", "polêmica", "insight valioso", "história curiosa", "aprendizado"]
pasta_entrada = "videos_brutos"
pasta_saida = "cortes_finalizados"
if not os.path.exists(pasta_saida): os.makedirs(pasta_saida)

def processar_arquivos():
    arquivos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith((".mp4", ".webm", ".mkv"))]
    
    for arquivo in arquivos:
        print(f"🎬 Editando: {arquivo}")
        caminho_video = os.path.join(pasta_entrada, arquivo)
        video = VideoFileClip(caminho_video)
        resultado = modelo_ia.transcribe(caminho_video, fp16=(device_gpu=="cuda"))
        
        # Passo 1: Identificar pontos de interesse
        pontos_interesse = []
        for seg in resultado['segments']:
            texto = seg['text'].strip()
            if len(texto) < 10: continue
            
            # --- O NOVO DETECTOR DE GARGALHADA ---
            texto_minusculo = texto.lower()
            tem_risada = "[risos]" in texto_minusculo or "(risos)" in texto_minusculo or "haha" in texto_minusculo or "kkk" in texto_minusculo

            res_sent = sentimento_ai(texto)[0]
            res_pontos = pontuador_ai(texto, candidate_labels=temas_relevantes)
            nota_rel = max(res_pontos['scores'])

            # Lógica de decisão (Lapidada)
            if (res_sent['score'] > MIN_SCORE_SENT) or (nota_rel > MIN_SCORE_PONTOS) or tem_risada:
                # Aplicamos o BUFFER (respiro) já na detecção
                pontos_interesse.append({
                    'start': max(0, seg['start'] - BUFFER_TIME),
                    'end': min(video.duration, seg['end'] + BUFFER_TIME)
                })

        # Passo 2: Mesclagem Agressiva (Transformar fragmentos em contexto)
        cortes_finais = []
        if pontos_interesse:
            atual = pontos_interesse[0]
            for i in range(1, len(pontos_interesse)):
                proximo = pontos_interesse[i]
                # Se o próximo ponto começa logo após o fim do atual (considerando o GAP)
                if proximo['start'] - atual['end'] < MAX_GAP_MERGE:
                    atual['end'] = max(atual['end'], proximo['end'])
                else:
                    # Só adiciona se o bloco final tiver a duração mínima
                    if (atual['end'] - atual['start']) >= MIN_DURATION:
                        cortes_finais.append(atual)
                    atual = proximo
            # Adiciona o último bloco se ele for longo o suficiente
            if (atual['end'] - atual['start']) >= MIN_DURATION:
                cortes_finais.append(atual)

        # Passo 3: Exportação
        for i, tempo in enumerate(cortes_finais, 1):
            duracao = tempo['end'] - tempo['start']
            nome_saida = f"corte_PRO{i}_{os.path.splitext(arquivo)[0]}.mp4"
            print(f"   ✨ Exportando: {nome_saida} ({duracao:.1f}s)")
            
            clip = video.subclipped(tempo['start'], tempo['end'])
            clip.write_videofile(
                os.path.join(pasta_saida, nome_saida),
                codec="libx264", audio_codec="aac",
                preset="ultrafast", logger=None
            )
        video.close()
    print("✅ Lapidação concluída!")

if __name__ == "__main__":
    processar_arquivos()