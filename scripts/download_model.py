import os
from TTS.utils.manage import ModelManager

# Define o diretório onde os modelos serão salvos
model_dir = os.path.join(os.getcwd(), 'model')
os.makedirs(model_dir, exist_ok=True)

# URLs dos arquivos do modelo XTTS v2.0
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# Define os caminhos completos para salvar os arquivos
DVAE_CHECKPOINT = os.path.join(model_dir, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(model_dir, os.path.basename(MEL_NORM_LINK))
TOKENIZER_FILE = os.path.join(model_dir, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(model_dir, os.path.basename(XTTS_CHECKPOINT_LINK))

# Baixa os arquivos do modelo XTTS v2.0 se eles não existirem
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Baixando os arquivos DVAE!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], model_dir, progress_bar=True)

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Baixando os arquivos do XTTS v2.0!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], model_dir, progress_bar=True
    )

print(f"Arquivos do modelo salvos em: {model_dir}")
