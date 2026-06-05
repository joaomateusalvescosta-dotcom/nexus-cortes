import whisper
from moviepy import VideoFileClip

# 1. IA OUVE O VÍDEO E PEGA OS TEMPOS
print("👂 Analisando áudio e tempos... aguarde.")
model = whisper.load_model("base")
# O segredo está aqui: o Whisper nos dá o início e fim de cada frase
result = model.transcribe("audio_da_ia.mp3", fp16=False)

# 2. VAMOS PEGAR O PRIMEIRO TRECHO (Para testar)
# O Whisper organiza tudo em 'segments' (segmentos)
primeiro_trecho = result['segments'][0]
inicio = primeiro_trecho['start']
fim = primeiro_trecho['end']
texto = primeiro_trecho['text']

print(f"\n🎬 Trecho detectado: '{texto}'")
print(f"⏱️ Tempo: de {inicio:.2f}s até {fim:.2f}s")

# 3. HORA DA TESOURA (MoviePy)
print("\n✂️ Cortando o vídeo original...")
video_original = VideoFileClip("meu_video.mp4")

# Criamos o corte baseado nos segundos que a IA achou
corte_final = video_original.subclipped(inicio, fim)

# 4. SALVANDO O RESULTADO
corte_final.write_videofile("meu_primeiro_corte_ia.mp4")

video_original.close()
print("\n✅ SUCESSO! O arquivo 'meu_primeiro_corte_ia.mp4' foi criado!")