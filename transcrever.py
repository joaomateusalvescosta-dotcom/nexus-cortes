import whisper

# 1. Carregamos o "cérebro" da IA (Modelo Base)
print("Carregando o modelo de IA... Isso pode levar um minuto.")
modelo = whisper.load_model("base")

# 2. Mandamos a IA ouvir o áudio que você criou hoje cedo
print("A IA está processando o áudio e escrevendo o texto...")
resultado = modelo.transcribe("audio_da_ia.mp3")

# 3. Mostramos o resultado final na tela
print("\n" + "="*30)
print("TEXTO TRANSCRITO:")
print(resultado["text"])
print("="*30)