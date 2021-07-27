# This file contains all class definitions used in visualization.py

class Token:
    def __init__(self, text, index) -> None:
        self.text = text
        self.index = index

        self.segment_id = -1
        self.mask = -1 # 0 = token can be in answer, 1 = token cannot be in answer
        self.is_special_token = False # True if token is [CLS] or [SEP]
        self.bert_id = -1

        self.label = ""

    def __repr__(self) -> str:
        return self.text

class ContextToken(Token):
    subtokens = []
    subtoken_index = -1

    def __repr__(self) -> str:
        return f"[{self.index}] {self.text}"

class ContextSubtoken(Token):
    def __init__(self, text, index, token) -> None:
        Token.__init__(self, text, index)
        self.token = token

    def __repr__(self) -> str:
        return self.text

class ContextSentenceToken:
    def __init__(self, text, index, position) -> None:
        self.text = text
        self.index = index # index in context
        self.position = position
        self.length = len(self.text)
    
    def __repr__(self) -> str:
        return self.text

class QuestionToken(Token):
    def __repr__(self) -> str:
        return self.text

class Answer:
    def __init__(self, text) -> None:
        self.text = text 
        self.length = len(self.text.split())

        self.index = -1
        self.sentence = None
        self.position = -1
        self.first_token = None
        self.end = -1
        self.last_token = None
        self.tokens = []
    
    def __repr__(self) -> str:
        return f"[{self.position}, {self.end}] {self.text}"

class SupportingFacts:
    def __init__(self) -> None:
        self.start = -1
        self.end = -1
        self.token_range = []
        self.tokens = []

class ContextWindow:
    def __init__(self, start, length) -> None:
        self.start = start
        self.length = length
        self.tokens = []

    def __repr__(self) -> str:
        return f"Window {self.start} -> {self.length}"

class ContextWindowResult:
    def __init__(self, window) -> None:
        self.window = window
        self.tokens = []
        self.cls_position = -1
        self.unpadded_length = -1
        self.bert_output = None
        self.bert_hidden_states = None
        self.final_predictions = None
        self.final_prediction = None

    def __repr__(self) -> str:
        return f"Result of window {self.window.start} -> {self.window.length}: {self.final_prediction.final_text} ({self.final_prediction.probability}%)"

class PaddingToken(Token):
    def __init__(self) -> None:
        Token.__init__(self, "", -1)
        self.bert_id = 0
        self.segment_id = 0
        self.mask = 1
        self.label = "Other"

    def __repr__(self) -> str:
        return "Padding"

class Prediction:
    def __init__(self, start_logit, end_logit) -> None:
        self.start = start_logit[0]
        self.start_logit = start_logit[1]
        self.end = end_logit[0]
        self.end_logit = end_logit[1]

        self.score = 0
        self.score_exp = 0

        self.probability = 0

        self.token = []
        self.tokens_string = ""
        self.context_tokens = []
        self.context_string = []

        self.final_string = "" #ToDo: remove ?

    def __repr__(self) -> str:
        return f"Prediction {self.start} -> {self.end}: {self.tokens_string}"

class TokenVector:
    def __init__(self, x, y, token) -> None:
        self.x = x
        self.y = y
        self.token = token

    def __repr__(self) -> str:
        return f"({self.x};{self.y}) {self.token}"
        