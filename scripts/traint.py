import os
import shutil
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Parâmetros de logging
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Defina o caminho onde os checkpoints serão salvos. Padrão: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# Define o prefixo da pasta a ser excluída
FOLDER_PREFIX = "GPT_XTTS_v2.0_LJSpeech"

# Remove diretórios antigos que começam com o prefixo especificado
##for item in os.listdir(OUT_PATH):
  ##  item_path = os.path.join(OUT_PATH, item)
  ##  if os.path.isdir(item_path) and item.startswith(FOLDER_PREFIX):
      ##  print(f"Removendo diretório antigo: {item_path}")
     ##   shutil.rmtree(item_path)

# Parâmetros de treinamento
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # para treinamento multi-GPU, defina como False
START_WITH_EVAL = True  # se True, iniciará com avaliação
BATCH_SIZE = 3  # defina aqui o tamanho do batch
GRAD_ACUMM_STEPS = 84  # defina aqui os passos de acumulação de gradiente
# Nota: recomendamos que BATCH_SIZE * GRAD_ACUMM_STEPS seja pelo menos 252 para um treinamento mais eficiente. Você pode aumentar/diminuir BATCH_SIZE e ajustar GRAD_ACUMM_STEPS conforme necessário.

# Defina aqui o dataset que você deseja usar para o fine-tuning.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path=r"C:\\Users\\Marx\\train_model\\",
    meta_file_train=r"wavs\\metadata.csv",
    language="pt-br",
)

# Adicione aqui as configurações dos datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Defina o caminho onde os arquivos do XTTS v2.0.1 serão baixados
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# Arquivos DVAE
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Defina o caminho para os arquivos baixados
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# Baixe os arquivos DVAE se necessário
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Baixando arquivos DVAE!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Baixe o checkpoint do XTTS v2.0 se necessário
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# Parâmetros de transfer learning do XTTS: Você precisa fornecer os caminhos do checkpoint do modelo XTTS que deseja ajustar.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # arquivo vocab.json
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # arquivo model.pth

# Baixe os arquivos do XTTS v2.0 se necessário
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Baixando arquivos do XTTS v2.0!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# Geração de frases para treinamento
SPEAKER_REFERENCE = [
    "wavs\wavs\chunk_0000.wav"  # referência de fala para ser usada nas frases de teste durante o treinamento
]
LANGUAGE = config_dataset.language

def main():
    try:
        # Inicializa os argumentos e a configuração
        model_args = GPTArgs(
            max_conditioning_length=132300,  # 6 segundos
            min_conditioning_length=66150,  # 3 segundos
            debug_loading_failures=False,
            max_wav_length=255995,  # ~11.6 segundos
            max_text_length=200,
            mel_norm_file=MEL_NORM_FILE,
            dvae_checkpoint=DVAE_CHECKPOINT,
            xtts_checkpoint=XTTS_CHECKPOINT,  # caminho do checkpoint do modelo que você deseja ajustar
            tokenizer_file=TOKENIZER_FILE,
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        # Define a configuração de áudio
        audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
        # Configuração dos parâmetros de treinamento
        config = GPTTrainerConfig(
            output_path=OUT_PATH,
            model_args=model_args,
            run_name=RUN_NAME,
            project_name=PROJECT_NAME,
            run_description="""
                Treinamento GPT XTTS
                """,
            dashboard_logger=DASHBOARD_LOGGER,
            logger_uri=LOGGER_URI,
            audio=audio_config,
            batch_size=BATCH_SIZE,
            batch_group_size=48,
            eval_batch_size=BATCH_SIZE,
            num_loader_workers=8,
            eval_split_max_size=256,
            eval_split_size=0.03125,  # Ajustado para o tamanho mínimo permitido
            print_step=50,
            plot_step=100,
            log_model_step=1000,
            save_step=10000,
            save_n_checkpoints=1,
            save_checkpoints=True,
            # target_loss="loss",
            print_eval=False,
            # Valores do otimizador como tortoise, implementação do pytorch com modificações para não aplicar WD aos parâmetros que não são pesos.
            optimizer="AdamW",
            optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
            lr=5e-06,  # taxa de aprendizado
            lr_scheduler="MultiStepLR",
            # Ajustado conforme o novo esquema de etapas
            lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
            test_sentences=[
                {
                    "text": "Demorei bastante tempo para desenvolver uma voz, e agora que a tenho, não vou ficar em silêncio.",
                    "speaker_wav": SPEAKER_REFERENCE,
                    "language": LANGUAGE,
                },
                {
                    "text": "Este bolo está ótimo. É tão delicioso e úmido.",
                    "speaker_wav": SPEAKER_REFERENCE,
                    "language": LANGUAGE,
                },
            ],
        )

        # Inicializa o modelo a partir da configuração
        model = GPTTrainer.init_from_config(config)

        # Carrega amostras de treinamento
        train_samples, eval_samples = load_tts_samples(
            DATASETS_CONFIG_LIST,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=0.03125,  # Ajuste o tamanho do conjunto de avaliação aqui
        )

        print("Train samples:", train_samples)
        print("Eval samples:", eval_samples)

        # Inicializa o treinador e começa o treinamento
        trainer = Trainer(
            TrainerArgs(
                restore_path=None,  # o checkpoint do xtts é restaurado via chave xtts_checkpoint, então não é necessário restaurar usando o parâmetro restore_path do Trainer
                skip_train_epoch=False,
                start_with_eval=START_WITH_EVAL,
                grad_accum_steps=GRAD_ACUMM_STEPS,
            ),
            config,
            output_path=OUT_PATH,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        trainer.fit()

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
