import os
from TTS.api import TTS

# Caminho para a pasta do modelo
model_dir = os.path.join(os.getcwd(), 'model')

# Caminhos completos para os arquivos do modelo
DVAE_CHECKPOINT = os.path.join(model_dir, 'dvae.pth')
MEL_NORM_FILE = os.path.join(model_dir, 'mel_stats.pth')
TOKENIZER_FILE = os.path.join(model_dir, 'vocab.json')
XTTS_CHECKPOINT = os.path.join(model_dir, 'model.pth')

# Caminho para o dataset de treinamento
dataset_path = "D:/projetos/tts/wavs/metadata.csv"

# Configura o TTS com o modelo XTTS v2.0 e os arquivos necessários
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
          checkpoint_path=XTTS_CHECKPOINT,
          dvae_path=DVAE_CHECKPOINT,
          mel_stats_path=MEL_NORM_FILE,
          tokenizer_path=TOKENIZER_FILE)

# Realiza a adaptação de voz
tts.finetune(dataset_path=dataset_path, output_path=model_dir)

print("Adaptação de voz concluída!")
