# Passo 1: Importamos a ferramenta que mexe com vídeos
from moviepy import VideoFileClip

# Passo 2: Dizemos ao Python qual vídeo ele deve abrir
# Certifique-se que o nome do seu arquivo é exatamente 'meu_video.mp4'
video = VideoFileClip("meu_video.mp4")

# Passo 3: Mandamos ele extrair o áudio e salvar com um novo nome
print("Iniciando a extração do áudio... aguarde um momento.")
video.audio.write_audiofile("audio_da_ia.mp3")

# Passo 4: Fechamos o arquivo para liberar a memória do computador
video.close()

print("Pronto! O arquivo 'audio_da_ia.mp3' foi criado com sucesso.")