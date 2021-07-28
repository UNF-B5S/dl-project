from typing import Tuple, List

import os
import logging
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig

from data_utils import QASample, SquadExample, QAInputFeatures, RawResult, read_squad_example, \
    convert_qa_example_to_features, parse_prediction

logging.basicConfig(format="%(asctime)-15s %(message)s", level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class BertQAModel:
    def __init__(self, model_path: str, model_type: str, lower_case: bool, cache_dir: str, device: str = "cpu"):
        self.model_path = model_path
        self.model_type = model_type
        self.lower_case = lower_case
        self.cache_dir = cache_dir
        self.device = device

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        # Load a pretrained model that has been fine-tuned
        config = BertConfig.from_pretrained(self.model_type, output_hidden_states=True, cache_dir=self.cache_dir)

        pretrained_weights = torch.load(self.model_path, map_location=torch.device(self.device))
        model = BertForQuestionAnswering.from_pretrained(self.model_type,
                                                         state_dict=pretrained_weights,
                                                         config=config,
                                                         cache_dir=self.cache_dir)
        return model

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained(self.model_type, cache_dir=self.cache_dir, do_lower_case=self.lower_case)

    def tokenize_and_predict(self, input_sample: QASample) -> Tuple:
        squad_formatted_sample: SquadExample = read_squad_example(input_sample)

        input_features: QAInputFeatures = self.tokenize(squad_formatted_sample)

        with torch.no_grad():
            # ADDED: input_features is now a list. We do the BERT-processing for each item from this list
            # ADDED: we save the window where the prediction has the hishest probability for visualization
            max_probability = 0.0
            final_prediction = None
            final_hidden_states = None
            final_window = None
            for x in input_features:
                inputs = {'input_ids': x.input_ids,
                        'attention_mask': x.input_mask,
                        'token_type_ids': x.segment_ids
                        }

                # Make Prediction
                output: Tuple = self.model(**inputs)  # output format: start_logits, end_logits, hidden_states

                # Parse Prediction
                prediction, hidden_states = self.parse_model_output(output, squad_formatted_sample, x)

                # ADDED: If this window is the best one so far (in terms of probability of prediction) we save it
                if prediction["probability"] > max_probability:
                    max_probability = prediction["probability"]
                    final_prediction = prediction
                    final_hidden_states = hidden_states
                    final_window = x

            logger.info("Predicted Answer: {}".format(prediction["text"]))
            logger.info("Start token: {}, End token: {}".format(prediction["start_index"], prediction["end_index"]))

            return final_prediction, final_hidden_states, final_window

    def tokenize(self, input_sample: SquadExample) -> QAInputFeatures:
        features = convert_qa_example_to_features(example=input_sample,
                                                  tokenizer=self.tokenizer,
                                                  max_seq_length=384,
                                                  doc_stride=128,
                                                  max_query_length=64,
                                                  is_training=False)

        # ADDED: The function now returns a list of feature-sets, so we have to create the tensors for each
        for x in features:
            x.input_ids = torch.tensor([x.input_ids], dtype=torch.long)
            x.input_mask = torch.tensor([x.input_mask], dtype=torch.long)
            x.segment_ids = torch.tensor([x.segment_ids], dtype=torch.long)
            x.cls_index = torch.tensor([x.cls_index], dtype=torch.long)
            x.p_mask = torch.tensor([x.p_mask], dtype=torch.float)

        return features

    @staticmethod
    def parse_model_output(output: Tuple, sample: SquadExample, features: QAInputFeatures) -> Tuple:
        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        result: RawResult = RawResult(unique_id=1,
                                      start_logits=to_list(output[0][0]),
                                      end_logits=to_list(output[1][0]))

        nbest_predictions: List = parse_prediction(sample, features, result)

        return nbest_predictions[0], output[2]  # top prediction, hidden states
