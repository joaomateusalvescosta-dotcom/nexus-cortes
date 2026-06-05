from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def identificar_melhores_partes(texto):
    # 1. Dividimos o texto em frases (usando o ponto como referência)
    # Como o Whisper as vezes não coloca ponto, vamos dividir por espaços a cada 10 palavras para simular frases
    palavras = texto.split()
    frases = [" ".join(palavras[i:i+15]) for i in range(0, len(palavras), 15)]
    
    # 2. Criamos um "pontuador" de relevância (TF-IDF)
    vetorizador = TfidfVectorizer()
    matriz = vetorizador.fit_transform(frases)
    
    # 3. Somamos a importância de cada frase
    pontuacoes = np.asarray(matriz.sum(axis=1)).flatten()
    
    # 4. Pegamos o índice da frase com maior pontuação
    melhor_frase_idx = np.argmax(pontuacoes)
    
    return frases[melhor_frase_idx]

# --- TESTE ---
# Aqui vamos simular que pegamos o resultado do Whisper
meu_texto_da_ia = "Vou começar a andar mais com o CF e o carro. O sucesso depende da constância e do esforço diário. Se você não focar no seu objetivo, nada vai acontecer."

melhor_corte = identificar_melhores_partes(meu_texto_da_ia)

print("\n" + "="*30)
print("🤖 ANÁLISE DA IA DE CORTES:")
print(f"Frase com maior potencial de viralizar: \n' {melhor_corte} '")
print("="*30)