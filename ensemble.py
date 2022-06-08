from Config import *
from utils import *
from Dataset import *
from Network import *
import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from tqdm.auto import tqdm

# Recovery class mapping for task 2.
classes = {
    0: 0,   # Not Mentioned(Don't care)
    1: -1,  # Negative
    2: 0,   # Neutral
    3: 1,   # Positive
}

if __name__ == "__main__":
    args = TrainConfig().parse_args()

    if args.task == 1:
        model_path_pretrained = {
            "best_model_task1_erlangshen_roberta_110m_sentiment.pt": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
            "best_model_task1_erlangshen_roberta_330m_sentiment.pt": "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
            "best_model_task1_erlangshen_roberta_110m_nli.pt": "IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
            "best_model_task1_erlangshen_roberta_110m_similarity.pt": "IDEA-CCNL/Erlangshen-Roberta-110M-Similarity",
            "best_model_task1_chinese_roberta_wwm_ext_large.pt": "hfl/chinese-roberta-wwm-ext-large",
            "best_model_task1_chinese_macbert_large.pt": "hfl/chinese-macbert-large",
            # "best_model_task1_ernie_gram_zh.pt": "nghuyong/ernie-gram-zh",
            "best_model_task1_xlm_roberta_base_snli_mnli_anli_xnli.pt": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
            "best_model_task1_bart_base_chinese.pt": "fnlp/bart-base-chinese",
            "best_model_task1_bart_large_chinese.pt": "fnlp/bart-large-chinese",
        }
    else:
        model_path_pretrained = {
            "best_model_task2_erlangshen_roberta_110m_sentiment.pt": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
            "best_model_task2_erlangshen_roberta_110m_nli.pt": "IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
            "best_model_task2_erlangshen_roberta_110m_similarity.pt": "IDEA-CCNL/Erlangshen-Roberta-110M-Similarity",
            "best_model_task2_chinese_roberta_wwm_ext_large.pt": "hfl/chinese-roberta-wwm-ext-large",
            "best_model_task2_chinese_macbert_large.pt": "hfl/chinese-macbert-large",
            # "best_model_task2_ernie_gram_zh.pt": "nghuyong/ernie-gram-zh",
            "best_model_task2_xlm_roberta_base_snli_mnli_anli_xnli.pt": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
            "best_model_task2_bart_base_chinese.pt": "fnlp/bart-base-chinese",
            "best_model_task2_bart_large_chinese.pt": "fnlp/bart-large-chinese",
        }

    # Fixed seed
    same_seeds(args.seed)

    # Device
    device = args.device
    # if args.fp16_training:
    #     accelerator, device = fp16_accelerator()

    # Reading files
    train_ids, train_reviews, train_sentiments,  = read_data("./data/train.csv", args.task)
    dev_ids, dev_reviews, dev_sentiments = read_data("./data/dev.csv", args.task)
    test_ids, test_reviews, test_sentiments = read_data("./data/test.csv", args.task)

    # Preprocessing
    replace_symbols = ["\u3000", "\\n", "\\", " "]
    replace_symbols = dict(zip(replace_symbols, [""] * len(replace_symbols)))
    train_reviews_clean = processing(train_reviews, replace_symbols)
    dev_reviews_clean = processing(dev_reviews, replace_symbols)
    test_reviews_clean = processing(test_reviews, replace_symbols)

    # Label index mapping
    aspects = [aspect for aspect in train_sentiments.columns]
    id2aspect = {idx:aspect for idx, aspect in enumerate(aspects)}
    aspect2id = {aspect:idx for idx, aspect in enumerate(aspects)}

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    # Dataset
    train_dataset, dev_dataset, test_dataset = get_dataset(
                                                train_reviews_clean,        # Training reviews.
                                                train_sentiments,           # Training labeled data.
                                                dev_reviews_clean,          # Deviation reviews.
                                                dev_sentiments,             # Deviation labeled data.
                                                test_reviews_clean,         # Testing reviews.
                                                test_sentiments,            # Testing labeled data(Empty).
                                                tokenizer,                  # Pretrained tokenizer.
                                                args                        # Config file.
                                            )

    # DataLoader
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers*2)
    dev_loader    = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # Load models
    models = []
    for key, value in model_path_pretrained.items():
        if args.task == 1:
            num_classes = len(aspects)
            model = MultilabelClassifier(num_classes, value).to(device)
        else:
            num_classes = len(classes)
            model = SentimentClassifier(num_classes, value, aspects).to(device)
        model_path = os.path.join(args.models_dir, key)
        model.load_state_dict(torch.load(model_path))
        models.append(model)

    # Validation
    if args.task == 1:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print("Evaluating Dev Set ...")
    with torch.no_grad():
        dev_loss, dev_acc = 0, 0
        y_pred, y_true = [], []
        for i, data in enumerate(tqdm(dev_loader)):
            ids, mask, targets = data["ids"], data["mask"], data["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            if args.task == 1:
                targets = targets.to(device, dtype=torch.float)
                # Ensemble the output logits
                ensemble_outputs = []
            else:
                targets = targets.to(device, dtype=torch.long).transpose(0, 1)
                # Ensemble the output logits
                ensemble_outputs = {}
                for aspect in aspects:
                    ensemble_outputs[aspect] = []

            batch_size = mask.size(0)
            # Append output logits
            for model in models:
                model.eval()
                outputs = model(input_ids=ids, attention_mask=mask)
                # Model is SequenceClassification object, then get the logits.
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits

                if args.task == 1:
                    ensemble_outputs.append(outputs)
                else:
                    for aspect in aspects:
                        ensemble_outputs[aspect].append(outputs[aspect])

            # Soft voting ensemble outputs
            if args.task == 1:
                ensemble_outputs = torch.cat(ensemble_outputs, dim=0)
                ensemble_outputs = ensemble_outputs.reshape(len(models), batch_size, -1).mean(dim=0)
                y_pred.append(ensemble_outputs) # Append ensemble predictions.
                y_true.append(targets)          # Append true labels.
                loss = loss_fn(outputs, targets.float())
            else:
                for aspect in aspects:
                    ensemble_outputs[aspect] = torch.cat(ensemble_outputs[aspect], dim=0)
                    ensemble_outputs[aspect] = ensemble_outputs[aspect].reshape(len(models), batch_size, -1).mean(dim=0)

                loss, acc = criterion(loss_fn, ensemble_outputs, targets, aspect2id)
                dev_acc += acc

            dev_loss += loss.detach().item()

        # Validation result
        if args.task == 1:
            y_pred = torch.cat(y_pred, axis=0)          # Concatenate the list of prediction batch.
            y_true = torch.cat(y_true, axis=0)          # Concatenate the list of true label batch.
            dev_f1, dev_acc = multi_label_metrics(y_pred, y_true)
            print(f"Validation | loss = {dev_loss / len(dev_loader):.3f}, acc = {dev_acc:.3f}, f1 = {dev_f1:.3f}")
        else:
            print(f"Validation | loss = {dev_loss / len(dev_loader):.3f}, acc = {dev_acc / len(dev_loader):.3f}")


    print("Evaluating Test Set ...")
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            ids, mask = data["ids"], data["mask"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            if args.task == 1:
                # Ensemble the output logits
                ensemble_outputs = []
            else:
                # Ensemble the output logits
                ensemble_outputs = {}
                for aspect in aspects:
                    ensemble_outputs[aspect] = []

            batch_size = mask.size(0)
            # Append output logits
            for model in models:
                model.eval()
                outputs = model(input_ids=ids, attention_mask=mask)
                # Model is SequenceClassification object, then get the logits.
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits

                if args.task == 1:
                    ensemble_outputs.append(outputs)
                else:
                    for aspect in aspects:
                        ensemble_outputs[aspect].append(outputs[aspect])

            # Soft voting ensemble outputs
            if args.task == 1:
                ensemble_outputs = torch.cat(ensemble_outputs, dim=0)
                ensemble_outputs = ensemble_outputs.reshape(len(models), batch_size, -1).mean(dim=0)
                results += evaluate_task1(ensemble_outputs, threshold=0.5)
            else:
                for aspect in aspects:
                    ensemble_outputs[aspect] = torch.cat(ensemble_outputs[aspect], dim=0)
                    ensemble_outputs[aspect] = ensemble_outputs[aspect].reshape(len(models), batch_size, -1).mean(dim=0)
                results += evaluate_task2(ensemble_outputs, batch_size, len(aspects), classes)

    result_file = f"prediction_task{args.task}_ensemble.csv"
    indices = [f"{id}-{idx+1}" for id in test_ids for idx in id2aspect.keys()]
    # Write file
    if args.task == 1:
        with open(result_file, 'w') as f:
            f.write(f"id-#aspect,predicted\n")
            for idx, result in zip(indices, results):
                result = idx.split('-')[0] if result == 1 else ""
                f.write(f"{idx},{result}\n")
    else:
        with open(result_file, 'w') as f:
            f.write(f"id-#aspect,sentiment\n")
            for idx, result in zip(indices, results):
                f.write(f"{idx},{result}\n")

    print(f"Completed! Result is in {result_file}")