# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

import argparse
import datetime
import json
import math
import matplotlib.pyplot as plt
import nltk
import os
from numpy import append
import string
import torch
import transformers
import sklearn.decomposition

from visualization_classes import *

# Step 0) Set up (cmd line args, directories etc.)
parser = argparse.ArgumentParser()
parser.add_argument("-sample", required=True)
parser.add_argument("-model", required=True)
parser.add_argument("-type", required=False, default="bert-base-uncased")
parser.add_argument("-title", required=False, default="")
parser.add_argument("-format", required=False, default="png", choices=["png","pdf","svg","jpg"])
parser.add_argument('--no-legend', dest='legend', action='store_false')

args = parser.parse_args()

cache = ".\cache"
output = os.path.join(".\output", args.title if args.title != "" else datetime.datetime.now().strftime("%Y-%m-%d--%H-%M"))
if not os.path.exists(output):
    os.makedirs(output)

print(f"The results will be saved to: {os.path.abspath(output)}")

# Step 1) Load the sample from the file
with open(args.sample) as sample_file:
    sample_json = json.load(sample_file)
    
# Print info about loaded sample
print("# Contents of passed sample:")
print(f" Question: {sample_json['question']}")
print(f" Answer: {sample_json['answer']}")
print(f" Context: {sample_json['context']}")

# Preprocessing of the sample
sample_context_tokens = []
for i, token in enumerate(sample_json['context'].split()):
    sample_context_tokens.append(ContextToken(token, i))

sample_context_sentences = []
current_position = 0
for i, sentence in enumerate(nltk.sent_tokenize(sample_json['context'])): # calling .split(.) does not work properly with abbreviations like "e.g."
    tok = ContextSentenceToken(sentence, i, current_position)
    sample_context_sentences.append(tok)
    current_position += tok.length

sample_answer = Answer(sample_json['answer'])

sample_answer.index = sample_json['context'].lower().find(sample_json['answer'].lower())                                    # position within context
sample_answer.length = len(sample_json['answer'].split())                                                                   # no of words in answer
sample_answer.sentence = next(s for s in sample_context_sentences if sample_json['answer'].lower() in s.text.lower())       # sentence of context conaining the answer
sample_answer.position = next(i for i,x in enumerate(sample_context_tokens) if x.text.translate(str.maketrans('', '', string.punctuation)) == sample_json["answer"]) # position within tokens
sample_answer.first_token = sample_context_tokens[sample_answer.position]
sample_answer.end = sample_answer.position + sample_answer.length - 1                                                       # end position within tokens
sample_answer.last_token = sample_context_tokens[sample_answer.end]

for i in range(sample_answer.position, sample_answer.end+1):
    sample_answer.tokens.append(sample_context_tokens[i])

sample_question_tokens = []
for i, token in enumerate(sample_json['question'].split()):
    sample_question_tokens.append(QuestionToken(token, i))

supporting_facts = SupportingFacts()
supporting_facts.start = sample_answer.sentence.position
supporting_facts.end = sample_answer.sentence.position + sample_answer.sentence.length

for x in [supporting_facts.start, supporting_facts.end]:
    length_sum = 0
    for k, t in enumerate(sample_context_tokens):
        length_sum += len(t.text)
        if length_sum >= x:
            supporting_facts.token_range.append(k)
            break
        else:
            length_sum += 1

supporting_facts.tokens = sample_context_tokens[supporting_facts.token_range[0]:(supporting_facts.token_range[1]+1)]

# Step 2) Load the model and tokenizer
print("# Loading model and tokenizer")
weights = torch.load(args.model, map_location=torch.device('cpu'))
bert_config = transformers.BertConfig.from_pretrained(args.type, output_hidden_states=True, cache_dir=cache)
bert_model = transformers.BertForQuestionAnswering.from_pretrained(args.type, state_dict=weights, config=bert_config, cache_dir=cache)
bert_tokenizer = transformers.BertTokenizer.from_pretrained(args.type, cache_dir=cache) # ToDo: do_lower_case
bert_max_seq_length = 384 # ToDo: max. 384 immer bei BERT?

# Step 3) Processing
print("# Processing data")
question_tokens = []
for i, token in enumerate(bert_tokenizer.tokenize(sample_json['question'])):
    question_tokens.append(QuestionToken(token, i))

# ToDo: max_query_length?

# Tokenize each word to put it into BERT's format
subtokens_count = 0
for i, token in enumerate(sample_context_tokens):
    token.subtokens = []
    for k, subtoken in enumerate(bert_tokenizer.tokenize(token.text)):
        token.subtokens.append(ContextSubtoken(subtoken, k, token))
    
    token.subtoken_index = subtokens_count
    subtokens_count += len(token.subtokens)

context_all_subtokens = []
for token in sample_context_tokens:
    for subtoken in token.subtokens:
        context_all_subtokens.append(subtoken)

# ToDo: _improve_answer_span ?

max_tokens = bert_max_seq_length - len(question_tokens) - 3 # 3 => 1x[CLS], 2x[SEP] need to be added to the token
context_windows = []
start = 0
step_size = 128
while start < len(context_all_subtokens):
    length = len(context_all_subtokens) - start
    if length > max_tokens:
        length = max_tokens

    context_window = ContextWindow(start, length)
    context_window.tokens = context_all_subtokens[start:(start+length)]
    context_windows.append(context_window)

    if start + length == len(context_all_subtokens):
        break
    else:
        start += min(length, step_size)

window_results = []
for (window_index, window) in enumerate(context_windows):
    result = ContextWindowResult(window)
    window_results.append(result)

    # List of tokens that will be passed to BERT
    result.tokens = []

    # Add [CLS] token at the beginning
    result.cls_position = len(result.tokens)
    cls_token = Token("[CLS]", -1)
    cls_token.segment_id = 0
    cls_token.mask = 0
    cls_token.is_special_token = True
    cls_token.label = "Other"
    result.tokens.append(cls_token)

    # Add tokens from question
    for question_token in question_tokens:
        question_token.segment_id = 0
        question_token.mask = 1
        question_token.label = "Question"
        result.tokens.append(question_token)

    # Add 1st [SEP] token
    sep1_token = Token("[SEP]", -1)
    sep1_token.segment_id = 0
    sep1_token.mask = 1
    sep1_token.is_special_token = True
    sep1_token.label = "Other"
    result.tokens.append(sep1_token)

    # Add tokens from this window
    for token in window.tokens:
        token.segment_id = 1
        token.mask = 0
        result.tokens.append(token)

        if token.token in supporting_facts.tokens:
            token.label = "Supporting_Fact"
        else:
            token.label = "Other"

    # Add 2nd [SEP] token
    sep2_token = Token("[SEP]", -1)
    sep2_token.segment_id = 1
    sep2_token.mask = 1
    sep2_token.is_special_token = True
    sep2_token.label = "Other"
    result.tokens.append(sep2_token)

    # Get BERT-IDs and add to tokens
    for id_index, id in enumerate(bert_tokenizer.convert_tokens_to_ids([x.text for x in result.tokens])):
        result.tokens[id_index].bert_id = id

    # Add padding
    result.unpadded_length = len(result.tokens)
    padding_diff = bert_max_seq_length - len(result.tokens)
    for i in range(padding_diff):
        result.tokens.append(PaddingToken())

    # Create torch tensors for BERT
    tensor_token_ids = torch.tensor([[x.bert_id for x in result.tokens]], dtype=torch.long)
    tensor_segment_ids = torch.tensor([[x.segment_id for x in result.tokens]], dtype=torch.long)
    tensor_mask = torch.tensor([[x.mask for x in result.tokens]], dtype=torch.long)
    tensor_padding_mask = torch.tensor([[0 if type(x) is PaddingToken else 1 for x in result.tokens]], dtype=torch.long)
    tensor_cls_position = torch.tensor([result.cls_position], dtype=torch.long)

    # BERT
    print("# BERT processing")
    with torch.no_grad():
        result.bert_output = bert_model(**{"input_ids":tensor_token_ids, "attention_mask":tensor_padding_mask, "token_type_ids":tensor_segment_ids})

    result.bert_hidden_states = result.bert_output[2]

    # https://huggingface.co/transformers/main_classes/output.html#questionansweringmodeloutput
    bert_output_start_logits = sorted(enumerate(result.bert_output[0][0].detach().cpu().tolist()), key=lambda x: x[1], reverse=True)[:20]
    bert_output_end_logits = sorted(enumerate(result.bert_output[1][0].detach().cpu().tolist()), key=lambda x: x[1], reverse=True)[:20]

    predictions = []
    for start_logit in bert_output_start_logits:
        for end_logit in bert_output_end_logits:
            # ToDo: max_answer_length = 30 ???
            if end_logit[0] < start_logit[0] or start_logit[0] >= len(result.tokens) or end_logit[0] >= len(result.tokens) or (end_logit[0] - start_logit[0] +1) > 30:
                continue

            prediction = Prediction(start_logit, end_logit)
            prediction.score = prediction.start_logit + prediction.end_logit

            prediction.tokens = result.tokens[prediction.start:(prediction.end+1)]
            prediction.tokens_string = " ".join([x.text for x in prediction.tokens]).replace(" ##", "##").replace("##", "").strip()
            prediction.tokens_string = " ".join(prediction.tokens_string.split()) # turns any multiple whitespaces into a single whitespace

            for t in prediction.tokens:
                if not t.is_special_token:
                    if type(t) is ContextSubtoken:
                        prediction.context_tokens.append(t.token)
                    elif type(t) is QuestionToken:
                        prediction.context_tokens.append(t)
            prediction.context_string = " ".join([x.text for x in prediction.context_tokens])

            predictions.append(prediction)

    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

    limit = 20
    result.final_predictions = []
    for prediction in predictions:
        if len(result.final_predictions) == limit:
            break
        
        if prediction.start > 0:
            prediction.final_text = prediction.tokens_string

            if any(x.final_text == prediction.final_text for x in result.final_predictions):
                continue
            else:
                result.final_predictions.append(prediction)

    # Compute softmax for each prediction
    all_scores = [x.score for x in result.final_predictions]
    max_score = max(all_scores)

    for prediction in result.final_predictions:
        prediction.score_exp = math.exp(prediction.score - max_score)

    total_sum = sum([x.score_exp for x in result.final_predictions])

    for prediction in result.final_predictions:
        prediction.probability = prediction.score_exp / total_sum

    result.final_prediction = result.final_predictions[0]

    for token in result.final_prediction.tokens:
        token.label = "Prediction"

# Step 4) Create the visualization of the best result
print("# Printing layers")
window_results = sorted(window_results, key=lambda x: x.final_prediction.probability, reverse=True)
best_window_result = window_results[0]

if args.title != "":
    plot_title = f"{args.title} - "
else:
    plot_title = ""

colors = {"Prediction":"#b04335", "Question":"#327da6", "Supporting_Fact":"#46ab7c", "Other":"#4f4f4f"}
labels = {"Prediction":"Prediction", "Question":"Question", "Supporting_Fact":"Supporting Fact", "Other":"Other"}

for layer_index, layer in enumerate(best_window_result.bert_hidden_states):
    tokens_unpadded = layer[0][:best_window_result.unpadded_length]
    pca_reduction = sklearn.decomposition.PCA(n_components=2)
    pca_reduction_transformed = pca_reduction.fit_transform(tokens_unpadded).transpose()

    token_vectors = []
    for token_index, token_value in enumerate(pca_reduction_transformed[0]):
        token_vectors.append(TokenVector(token_value, pca_reduction_transformed[1][token_index], best_window_result.tokens[token_index]))

    used_labels = []
    for vector in token_vectors:
        if vector.token.is_special_token:
            continue

        selected_label = labels[vector.token.label]
        if selected_label in used_labels:
            selected_label = ""
        else:
            used_labels.append(selected_label)

        plt.scatter(vector.x, vector.y, c=colors[vector.token.label], label=selected_label)
        plt.text(vector.x, vector.y, vector.token.text)

    layer_title = f"{plot_title}Layer #{layer_index}"
    plt.title(layer_title)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    if args.legend:
        plt.legend()

    layer_plot_path = os.path.join(output, f"layer-{layer_index}.{args.format}")
    plt.savefig(layer_plot_path)
    plt.clf()

    print(f" Printed layer #{layer_index+1} of {len(best_window_result.bert_hidden_states)}")

# Finished
print(f"# Finished. Check the folder {output} for the results!")
