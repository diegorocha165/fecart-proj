# voice_cloning.py
# Demonstração educacional de clonagem de voz com Tortoise-TTS

import torch
import torchaudio
import numpy as np
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import sounddevice as sd
import time
import warnings
warnings.filterwarnings("ignore")

class VoiceCloningDemo:
    def __init__(self):
        print("Inicializando sistema de clonagem de voz...")
        print("Isso pode levar alguns minutos na primeira execução...")
        
        # Inicializar o modelo TTS
        self.tts = TextToSpeech()
        
        # Texto para conscientização sobre IA
        self.texto_conscientizacao = """
        Atenção! Esta é uma demonstração de como a inteligência artificial pode clonar vozes. 
        Tecnologias como esta representam avanços incríveis, mas também trazem sérios riscos.
        Com apenas alguns segundos da sua voz, é possível criar áudios falsos que soam exatamente como você.
        Isso pode ser usado para golpes, desinformação e violação de privacidade.
        Esteja sempre ciente desses perigos e advogue por regulamentações éticas para o uso de IA.
        """
    
    def gravar_voz_referencia(self, duracao=10, taxa_amostragem=22050):
        """Grava uma amostra de voz para clonagem"""
        print(f"\n🎤 Grave sua voz por {duracao} segundos...")
        print("Preparando...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("Fale agora!")
        
        # Gravar áudio do microfone
        audio = sd.rec(int(duracao * taxa_amostragem), 
                      samplerate=taxa_amostragem, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        
        # Converter e salvar o áudio
        audio_tensor = torch.from_numpy(audio.T)
        torchaudio.save("minha_voz.wav", audio_tensor, taxa_amostragem)
        print("✅ Voz gravada com sucesso!")
        
        return "minha_voz.wav"
    
    def clonar_voz(self, arquivo_voz):
        """Clona a voz e gera a mensagem de conscientização"""
        print("\n🔄 Processando e clonando sua voz...")
        print("Isso pode levar vários minutos...")
        
        # Carregar a voz de referência
        amostras_voz, latentes = load_voice(arquivo_voz)
        
        # Gerar áudio com a voz clonada
        audio_gerado = self.tts.tts_with_preset(
            self.texto_conscientizacao, 
            voice_samples=amostras_voz, 
            conditioning_latents=latentes,
            preset="standard"  # "standard", "high_quality", ou "ultra_fast"
        )
        
        # Salvar o áudio gerado
        torchaudio.save("voz_clonada.wav", audio_gerado.squeeze(0).cpu(), 24000)
        print("✅ Voz clonada com sucesso!")
        
        return audio_gerado
    
    def reproduzir_audio(self, audio_tensor, taxa_amostragem=24000):
        """Reproduz o áudio gerado"""
        audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
        print("\n🔊 Reproduzindo mensagem com sua voz clonada...")
        sd.play(audio_numpy, samplerate=taxa_amostragem)
        sd.wait()
    
    def demonstrar_perigos_ia(self):
        """Demonstração completa"""
        print("="*65)
        print("DEMONSTRAÇÃO DE CLONAGEM DE VOZ COM IA")
        print("="*65)
        
        # Gravar voz
        arquivo_voz = self.gravar_voz_referencia(duracao=8)
        
        # Clonar voz
        audio_clonado = self.clonar_voz(arquivo_voz)
        
        # Reproduzir resultado
        self.reproduzir_audio(audio_clonado)
        
        # Mensagem final
        print("\n" + "="*65)
        print("CONCLUSÃO E ALERTAS:")
        print("- IAs podem clonar vozes com precisão assustadora")
        print("- Sua identidade vocal pode ser roubada e utilizada")
        print("- Desconfie de áudios suspeitos, mesmo que soem reais")
        print("- Exija verificação em múltiplos canais para assuntos importantes")
        print("="*65)

if __name__ == "__main__":
    # Verificar se há GPU disponível
    if torch.cuda.is_available():
        print("✅ GPU detectada - processamento acelerado")
        device = "cuda"
    else:
        print("⚠️  Usando CPU - processamento será mais lento")
        device = "cpu"
    
    # Executar demonstração
    try:
        demo = VoiceCloningDemo()
        demo.demonstrar_perigos_ia()
    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        print("Verifique se todas as dependências foram instaladas corretamente")