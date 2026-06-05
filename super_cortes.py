import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1. FAZENDO A IA OUVIR (Igual fizemos antes)
print("👂 IA ouvindo o seu vídeo... aguarde.")
modelo = whisper.load_model("base")
resultado = modelo.transcribe("audio_da_ia.mp3", fp16=False)
texto_real_do_video = resultado["text"]

# 2. TRANSFORMANDO O TEXTO EM LISTA DE FRASES
# Vamos quebrar o texto do seu vídeo em pedaços para a IA analisar
palavras = texto_real_do_video.split()
frases = [" ".join(palavras[i:i+15]) for i in range(0, len(palavras), 15)]

# 3. ANALISANDO QUAL PARTE DO SEU VÍDEO É MELHOR
print("🧠 Analisando qual parte do SEU vídeo é a melhor...")
vetorizador = TfidfVectorizer()
matriz = vetorizador.fit_transform(frases)
pontuacoes = np.asarray(matriz.sum(axis=1)).flatten()

# 4. PEGANDO O RESULTADO REAL
melhor_frase_idx = np.argmax(pontuacoes)
print("\n" + "="*40)
print("🎯 RESULTADO DO SEU VÍDEO REAL:")
print(f"'{frases[melhor_frase_idx]}'")
print("="*40)