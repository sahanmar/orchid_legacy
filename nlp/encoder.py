import sys
from typing import List

from transformers import AutoTokenizer, AutoModel

from utils.util_types import EncodingType


ENCODER_MAPPER = {
    EncodingType.SpanBERT_base_cased.value: "SpanBERT/spanbert-base-cased",
    EncodingType.SpanBERT_large_cased.value: "SpanBERT/spanbert-large-cased",
}


class GeneralisedEncoder:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def load_model(model_type: EncodingType):

        if model_type.value not in ENCODER_MAPPER:
            print("The encoder is out of the menu..")
            sys.exit()

        encoder = ENCODER_MAPPER[model_type.value]

        model = AutoModel.from_pretrained(encoder)
        tokenizer = AutoTokenizer.from_pretrained(encoder)

        return GeneralisedEncoder(model, tokenizer)

    def encode_tokens(self, tokens: List[str]) -> Dict[List[torch]]:

        """
        This method is able to work with any tokenization. 
        Tokenized text is matched with PBE subtokens tokens obtained from transformers.
        The method returns a dict with encoded ids and 
        """

        ## return ids and encoded stuff. Dict[str, torch.Tensor]
        return self.tokenizer(tokens, is_split_into_words=True)
