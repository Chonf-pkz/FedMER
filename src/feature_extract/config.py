import torch
from transformers import AutoFeatureExtractor, BertTokenizer
from .model_encode import BERTEmbeddingModel, Wav2Vec2EmbeddingModel
PKL_DIR = "metadata"
OUTPUT_DIR = "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
AUDIO_PROCESSOR = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

TEXT_MODEL = BERTEmbeddingModel(embedding_dim=768, projection_dim=512).to(device)
AUDIO_MODEL = Wav2Vec2EmbeddingModel(embedding_dim=768, projection_dim=512).to(device)

TEXT_MODEL.eval()
AUDIO_MODEL.eval()