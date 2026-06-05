import os
import torch
import whisper
import yt_dlp
from transformers import pipeline
import subprocess

# --- ⚙️ CONSOLE DE CALIBRAÇÃO (LAPIDAÇÃO) ---
MIN_SCORE_SENT = 0.90      # Só passa emoção EXTREMA (nota > 85%)
MIN_SCORE_PONTOS = 0.90    # O assunto tem que ser MUITO relevante
MIN_DURATION = 15.0        
MAX_DURATION = 90.0        
MAX_GAP_MERGE = 2.0        
BUFFER_TIME = 1.0         
GORDURA_SNIPER = 3.0

# --- 1. HARDWARE ---
device_gpu = "cuda" if torch.cuda.is_available() else "cpu"
device_cpu = "cpu"

# --- 2. CARREGAMENTO DAS IAs ---
print("🧠 Ligando os motores de Inteligência Artificial...")

# Define os IDs corretos para o Transformers (0 = GPU, -1 = CPU)
id_gpu_transformers = 0 if torch.cuda.is_available() else -1

modelo_ia = whisper.load_model("tiny", device=device_gpu)
sentimento_ai = pipeline("text-classification", model="pysentimiento/bertweet-pt-sentiment", device=id_gpu_transformers)
pontuador_ai = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=id_gpu_transformers)

# --- 3. CONFIGURAÇÕES DE PASTAS ---
temas_relevantes = ["conselho", "polêmica", "insight valioso", "história curiosa", "aprendizado"]
pasta_audios = "audios_temp"
pasta_saida = "cortes_finalizados"

if not os.path.exists(pasta_audios): os.makedirs(pasta_audios)
if not os.path.exists(pasta_saida): os.makedirs(pasta_saida)

# --- 🎯 FUNÇÕES DO SNIPER ---
def sniper_extrair_audio(url_youtube):
    caminho_audio = os.path.join(pasta_audios, 'audio_alvo.mp3')
    
    # Se já existir um áudio velho de teste, apaga
    if os.path.exists(caminho_audio):
        os.remove(caminho_audio)
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(pasta_audios, 'audio_alvo.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True
    }
    print(f"\n🕵️‍♂️ SNIPER 1: Baixando áudio invisível do YouTube...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_youtube])
    return caminho_audio

def sniper_cortar_video(url_youtube, start, end, nome_arquivo):
    caminho_final = os.path.join(pasta_saida, nome_arquivo)
    caminho_temp = os.path.join(pasta_saida, f"temp_{nome_arquivo}")
    
    # --- PASSO 1: Baixar o formato Combo (Áudio e Vídeo colados e sincronizados) ---
    ydl_opts = {
        'format': 'b[ext=mp4]/b/best', 
        'outtmpl': caminho_temp,
        'download_ranges': lambda info_dict, ydl: [{'start_time': start, 'end_time': end}],
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'no_warnings': True
    }
    
    print(f"  🎯 SNIPER 2: Baixando vídeo combo estável ({start:.1f}s até {end:.1f}s)...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_youtube])
    except Exception as e:
        print(f"  ❌ Erro no download: {e}")
        return

    # --- PASSO 2: Fundo borrado usando CPU no modo MÁXIMA VELOCIDADE ---
    print(f"  🎨 SNIPER 3: Aplicando fundo borrado (Modo ULTRAFAST)...")
    
    filtro_fundo_borrado = (
        "[0:v]split[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg_rows];"
        "[fg]scale=1080:-2[fg_scaled];"
        "[bg_rows][fg_scaled]overlay=(W-w)/2:(H-h)/2"
    )

    comando_ffmpeg = [
        "ffmpeg", "-y", "-i", caminho_temp,
        "-vf", filtro_fundo_borrado,
        # 🔥 A MÁGICA DE VELOCIDADE: CPU rodando no limite da agilidade
        "-c:v", "libx264", "-preset", "ultrafast", 
        "-c:a", "aac", 
        caminho_final
    ]

    # Rodando SEM o DevNull. Agora vemos tudo o que o FFmpeg faz.
    subprocess.run(comando_ffmpeg)

    # Limpa o arquivo temporário
    if os.path.exists(caminho_temp):
        os.remove(caminho_temp)

    print(f"  ✅ Corte vertical finalizado com sucesso!")

# --- 🧠 CÉREBRO DO ROBÔ ---
def processar_url(url):
    print("\n==================================================")
    print("🚀 INICIANDO MÁQUINA DE CORTES AUTÔNOMA")
    print("==================================================")
    
    # 1. Puxa só o áudio
    caminho_audio = sniper_extrair_audio(url)
    
    # 2. IA escuta o áudio
    print(f"🧠 Whisper analisando o áudio...")
    resultado = modelo_ia.transcribe(caminho_audio, fp16=(device_gpu=="cuda"))
    
    # Descobre a duração total do vídeo baseado na última fala
    duracao_total = resultado['segments'][-1]['end'] if resultado['segments'] else 9999
    
    pontos_interesse = []
    print("⚖️ Juízes de Sentimento e Relevância avaliando falas...")
    
    for seg in resultado['segments']:
        texto = seg['text'].strip()
        if len(texto) < 10: continue
        
        texto_minusculo = texto.lower()
        tem_risada = "[risos]" in texto_minusculo or "(risos)" in texto_minusculo or "haha" in texto_minusculo or "kkk" in texto_minusculo

        res_sent = sentimento_ai(texto)[0]
        res_pontos = pontuador_ai(texto, candidate_labels=temas_relevantes)
        nota_rel = max(res_pontos['scores'])

        if (res_sent['score'] > MIN_SCORE_SENT) or (nota_rel > MIN_SCORE_PONTOS) or tem_risada:
            pontos_interesse.append({
                'start': max(0, seg['start'] - BUFFER_TIME),
                'end': min(duracao_total, seg['end'] + BUFFER_TIME)
            })

    # 3. Mesclagem Agressiva e Controle de Tempo
    cortes_finais = []
    if pontos_interesse:
        atual = pontos_interesse[0]
        for i in range(1, len(pontos_interesse)):
            proximo = pontos_interesse[i]
            
            # Condições para juntar os pedaços:
            # 1. Distância entre as falas é menor que o GAP
            # 2. O tamanho TOTAL do corte, se juntar, não passa do MAX_DURATION
            if (proximo['start'] - atual['end'] < MAX_GAP_MERGE) and ((proximo['end'] - atual['start']) <= MAX_DURATION):
                atual['end'] = max(atual['end'], proximo['end'])
            else:
                if (atual['end'] - atual['start']) >= MIN_DURATION:
                    cortes_finais.append(atual)
                atual = proximo
                
        if (atual['end'] - atual['start']) >= MIN_DURATION:
            cortes_finais.append(atual)

    # 4. Disparos do Sniper de Vídeo
    print(f"\n✨ Foram encontrados {len(cortes_finais)} cortes de ouro. Iniciando downloads cirúrgicos!")
    for i, tempo in enumerate(cortes_finais, 1):
        # Aplica a margem de segurança para os Keyframes
        start_seguro = max(0, tempo['start'] - GORDURA_SNIPER)
        end_seguro = min(duracao_total, tempo['end'] + GORDURA_SNIPER)
        
        nome_saida = f"corte_AUTONOMO_{i}.mp4"
        sniper_cortar_video(url, start_seguro, end_seguro, nome_saida)
        print(f"  ✅ {nome_saida} salvo na pasta!")

    # 5. Limpa a bagunça (apaga o MP3 temporário)
    if os.path.exists(caminho_audio):
        os.remove(caminho_audio)
        
    print("\n🏁 OPERAÇÃO CONCLUÍDA COM SUCESSO! Verifique sua pasta 'cortes_finalizados'.")

if __name__ == "__main__":
    # O bot pede a URL para você assim que o código roda!
    url_alvo = input("Cole a URL do vídeo ou podcast do YouTube: ").strip()
    
    if url_alvo:
        processar_url(url_alvo)
    else:
        print("URL inválida. Encerrando o sistema.")