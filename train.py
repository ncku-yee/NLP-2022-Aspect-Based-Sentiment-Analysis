from Config import *
from utils import *
from Dataset import *
from Network import *
import os
import numpy as np
import pandas as pd
import random
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer

# Recovery class mapping for task 2.
classes = {
    0: 0,   # Not Mentioned(Don't care)
    1: -1,  # Negative
    2: 0,   # Neutral
    3: 1,   # Positive
}

if __name__ == "__main__":

    args = TrainConfig().parse_args()

    # open log file to save log outputs
    log_f = open(f"{args.logs_path}", 'w')
    # Write Configuration into log file.
    for k, v in sorted(vars(args).items()):
        log(log_f, f"{k.ljust(20, ' ')}: {v}")

    # Fixed seed
    same_seeds(args.seed)

    # Device
    device = args.device
    if args.fp16_training:
        accelerator, device = fp16_accelerator()

    # Model directory
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)

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
    try:
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

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

    # Load model
    if args.task == 1:
        num_classes = len(aspects)
        model = MultilabelClassifier(num_classes, args.pretrained_model, drop_prob=args.drop_prob).to(device)
    else:
        num_classes = len(classes)
        model = SentimentClassifier(num_classes, args.pretrained_model, aspects, drop_prob=args.drop_prob).to(device)

    log(log_f, model)
    # if args.verbose:
    #     print(model)

    total_steps = len(train_loader) * args.num_epochs // args.accum_steps   # Total number of training steps


    if args.task == 1:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()

    if args.fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

    if args.is_train:
        print("Start Training ...")
        best_acc, best_f1 = 0, 0
        for epoch in range(args.num_epochs):
            # Training
            model, optimizer = train(
                    model,          # Model
                    optimizer,      # Opimizer
                    loss_fn,        # Loss function
                    accelerator,    # FP16 Accelerator
                    device,         # Current device
                    train_loader,   # Dataloader
                    aspect2id,      # aspect to index mapping
                    epoch,          # Current epoch
                    log_f,          # Logging file
                    args            # Arguments
                )
            # Validation
            if args.validation:
                    best_acc, best_f1 = valid(
                    model,          # Model
                    loss_fn,        # Loss function
                    device,         # Current device
                    dev_loader,     # Dataloader
                    aspect2id,      # aspect to index mapping
                    best_f1,        # Current best f1 score(Task 1)
                    best_acc,       # Current best accuracy(Task 2)
                    epoch,          # Current epoch
                    log_f,          # Logging file
                    args            # Arguments
                )
    # Just Validation
    elif args.validation:
        # Load model
        if args.task == 1:
            num_classes = len(aspects)
            model = MultilabelClassifier(num_classes, args.pretrained_model, drop_prob=args.drop_prob).to(device)
        else:
            num_classes = len(classes)
            model = SentimentClassifier(num_classes, args.pretrained_model, aspects, drop_prob=args.drop_prob).to(device)
        model.load_state_dict(torch.load(args.model_path))
        best_acc, best_f1 = valid(
                model,              # Model
                loss_fn,            # Loss function
                device,             # Current device
                dev_loader,         # Dataloader
                aspect2id,          # aspect to index mapping
                best_f1=np.inf,     # Current best f1 score(Task 1)
                best_acc=np.inf,    # Current best accuracy(Task 2)
                epoch=-1,           # Current epoch
                log_f=log_f,        # Logging file
                args=args           # Arguments
            )

    # Load model
    if args.task == 1:
        num_classes = len(aspects)
        model = MultilabelClassifier(num_classes, args.pretrained_model, drop_prob=args.drop_prob).to(device)
    else:
        num_classes = len(classes)
        model = SentimentClassifier(num_classes, args.pretrained_model, aspects, drop_prob=args.drop_prob).to(device)
    model.load_state_dict(torch.load(args.model_path))
    # Inference
    test(
        model,              # Model
        device,             # Current device
        test_loader,        # Dataloader
        test_ids,           # Reviews' id
        args,               # Arguments
        id2aspect=id2aspect,# index to aspect mapping
        classes=classes,    # Recovery class mapping
    )
