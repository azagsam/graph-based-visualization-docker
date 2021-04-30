from sentence_transformers import SentenceTransformer


class SentenceBERT:
    def __init__(self, model_dir):
        self.model = SentenceTransformer(model_name_or_path=model_dir)

    def encode_sentences(self, sentences, batch_size=32):
        return self.model.encode(sentences, convert_to_tensor=True, batch_size=batch_size)
