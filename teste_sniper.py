import os
import yt_dlp

def baixar_apenas_audio(url_youtube, pasta_destino="audios_brutos"):
    if not os.path.exists(pasta_destino): 
        os.makedirs(pasta_destino)
    
    # Configuração para extrair o melhor áudio
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(pasta_destino, 'audio_alvo.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False
    }
    
    print(f"\n🕵️‍♂️ Sniper mirando na URL: {url_youtube}")
    print("🎵 Baixando apenas a faixa de áudio...")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_youtube])
    
    print("\n✅ Missão cumprida! O áudio foi isolado.")

if __name__ == "__main__":
    # Coloque o link de um vídeo qualquer do YouTube aqui dentro das aspas!
    link_teste = "https://www.youtube.com/watch?v=VERH_lb5sRo" # (Link aqui)
    
    baixar_apenas_audio(link_teste)