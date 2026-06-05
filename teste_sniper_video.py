import os
import yt_dlp

def baixar_corte_cirurgico(url_youtube, tempo_inicio, tempo_fim, nome_saida="meu_corte_sniper", pasta_saida="cortes_sniper"):
    # Cria a pasta se não existir
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
        
    caminho_final = os.path.join(pasta_saida, f"{nome_saida}.mp4")
    
    # 🛡️ FILTRO ANTI-AV1: Força o codec H.264 (avc1) que o ffmpeg corta fácil
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best', 
        'outtmpl': caminho_final,
        
        # O GATILHO: Define exatamente onde começa e onde termina na nuvem!
        'download_ranges': lambda info_dict, ydl: [{
            'start_time': tempo_inicio, # Começa no segundo 10
            'end_time': tempo_fim       # Termina no segundo 30
        }],
        
        # Força o YouTube a cortar no milissegundo exato
        'force_keyframes_at_cuts': True, 
        'quiet': False
    }
    
    print(f"\n🎯 Atirador de Elite a postos para a URL: {url_youtube}")
    print(f"📹 Puxando APENAS o trecho de {tempo_inicio}s até {tempo_fim}s (Filtro H.264 ativado)...")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_youtube])
        
    print(f"\n✨ Operação concluída! Corte salvo em: {caminho_final}")

if __name__ == "__main__":
    # Coloque aqui o mesmo link que você usou no teste anterior
    link_teste = "https://www.youtube.com/watch?v=VERH_lb5sRo"
    
    # Vamos extrair apenas 20 segundos (do segundo 10 ao 30)
    baixar_corte_cirurgico(link_teste, tempo_inicio=10.0, tempo_fim=30.0)