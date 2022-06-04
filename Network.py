from transformers import BertModel, BertForSequenceClassification
from transformers import BartModel, BartForSequenceClassification
from transformers import AutoModel
import torch.nn as nn

# Support the pretrained model available on huggingface
# https://huggingface.co/models?language=zh&sort=downloads
MODELS = [
    'fnlp/bart-base-chinese',
    'fnlp/bart-large-chinese',
    'IDEA-CCNL/Erlangshen-Roberta-110M-NLI',
    'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment',
    'IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment',
    'IDEA-CCNL/Erlangshen-Roberta-110M-Similarity',
    'hfl/chinese-macbert-large',
    'hfl/chinese-roberta-wwm-ext-large',
    'symanto/xlm-roberta-base-snli-mnli-anli-xnli',
    'nghuyong/ernie-gram-zh',
    'ckiplab/bert-base-chinese-ws',
    'ckiplab/albert-base-chinese-ws',
]

# Model for Task 1.
class MultilabelClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model, drop_prob=0.1):
        super(MultilabelClassifier, self).__init__()
        # Check the pretrained model name.
        assert pretrained_model in MODELS, f"{pretrained_model} is not available."

        # Pretrained Model.
        self.pretrained_model = pretrained_model
        try:
            if 'bart' in self.pretrained_model.lower():
                self.bert = BartModel.from_pretrained(pretrained_model, return_dict=True)
            elif 'IDEA-CCNL' in self.pretrained_model:
                self.bert = BertForSequenceClassification.from_pretrained(pretrained_model)

                # Fine-tune the classifier's out_features.
                last_layer_name = next(reversed(self.bert._modules))    # Get the name of last layer of last module.
                # last module is nn.Sequential
                if isinstance(self.bert._modules[last_layer_name], nn.Sequential):
                    fc_in_features = self.bert._modules[last_layer_name][-1].in_features
                    self.bert._modules[last_layer_name][-1] = nn.Linear(fc_in_features, num_classes)
                # last module is nn.Linear
                else:
                    fc_in_features = self.bert._modules[last_layer_name].in_features
                    # You can add the MLP layer HERE.
                    self.bert._modules[last_layer_name] = nn.Sequential(
                        nn.Linear(fc_in_features, num_classes),
                    )
            else:
                self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        except:
            # Only support AutoModel
            self.bert = AutoModel.from_pretrained(pretrained_model, return_dict=True)

        # Hidden Representation dimension.
        d_model = self.bert.config.hidden_size

        # Feed Forward Network for multi-aspect.
        self.classifier = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        if 'IDEA-CCNL' in self.pretrained_model:
            x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            if 'bart' in self.pretrained_model.lower():
                x = self.bert(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
                x = x.mean(dim=1)   # Mean pooling
            else:
                x = self.bert(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
            x = self.classifier(x)
        return x

# Model for Task 2.
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model, aspects, drop_prob=0.3):
        super(SentimentClassifier, self).__init__()
        # Check the pretrained model name.
        assert pretrained_model in MODELS, f"{pretrained_model} is not available."

        # Pretrained Model.
        self.pretrained_model = pretrained_model
        try:
            if 'bart' in self.pretrained_model.lower():
                self.bert = BartModel.from_pretrained(pretrained_model, return_dict=True)
            else:
                self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        except:
            # Only support AutoModel
            self.bert = AutoModel.from_pretrained(pretrained_model, return_dict=True)

        # Hidden Representation dimension.
        d_model = self.bert.config.hidden_size

        # Feed Forward Network for each aspect.
        self.aspects = aspects
        self.classifier = nn.ModuleDict({})
        for aspect in aspects:
            self.classifier[aspect] = nn.Sequential(
                nn.Dropout(drop_prob),
                nn.Linear(d_model, num_classes),
            )

    def forward(self, input_ids, attention_mask):
        if 'bart' in self.pretrained_model.lower():
            x = self.bert(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
            x = x.mean(dim=1)   # Mean pooling
        else:
            x = self.bert(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
        # Output of each aspect.
        out = {}
        for aspect in self.aspects:
            out[aspect] = self.classifier[aspect](x)
        return out