import random
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from accelerate import Accelerator
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

""" Fix random seed for reproducibility """
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

""" FP16 Accelerator """
def fp16_accelerator():
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    return accelerator, device

""" Task 1 Metrics """
# Ref: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(preds, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(preds).detach().cpu()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels.cpu().detach().numpy()
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=np.unique(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    return f1_macro_average, accuracy

""" Task 2 Metrics """
def criterion(loss_fn, y_pred, y_true, aspect2id, topk=2):
    """
    Evaluation metrics.
    Arguments:
        loss_fn        : Loss function.
        y_pred(dict)   : Each aspects' prediction.
        y_true(2d list): Each aspects' ground truth.
        aspect2id(dict): Aspect to index table.
        topk(int)      : Get topk max logits.
    Returns:
        loss(float): Total loss.
        acc(float) : Total accuracy.
    """
    losses, acc = 0, 0
    focus_count = 0
    for key in y_pred.keys():
        loss = loss_fn(y_pred[key], y_true[aspect2id[key]])
        # acc += (y_pred[key].argmax(dim=-1) == y_true[aspect2id[key]]).float().sum().item()
        _, y_pred_topk = torch.topk(y_pred[key], topk, dim=-1)
        y_pred_topk = y_pred_topk.transpose(0, 1)
        for pred in y_pred_topk:
            acc += ((pred == y_true[aspect2id[key]]) & (y_true[aspect2id[key]] != 0)).float().sum().item()
        focus_count += (y_true[aspect2id[key]] != 0).float().sum().item()
        losses += loss
    acc = acc / focus_count if focus_count > 0 else 0       # Handle the zero division.
    return losses, acc

""" Training one epoch """
def train(
        model,          # Model
        optimizer,      # Opimizer
        loss_fn,        # Loss function
        accelerator,    # FP16 Accelerator
        device,         # Current device
        dataloader,     # Dataloader
        aspect2id,      # aspect to index mapping
        epoch,          # Current epoch
        args            # Arguments
    ):

    model.train()       # Training Stage
    step = 1            # Starting step
    train_loss, train_acc = 0, 0
    y_pred, y_true = [], []

    for batch_idx, data in enumerate(tqdm(dataloader)):
        ids, mask, targets = data["ids"], data["mask"], data["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        if args.task == 1:
            targets = targets.to(device, dtype=torch.float)
        else:
            targets = targets.to(device, dtype=torch.long).transpose(0, 1)

        outputs = model(input_ids=ids, attention_mask=mask)

        # Model is SequenceClassification object, then get the logits.
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits

        if args.task == 1:
            y_pred.append(outputs)      # Append predictions.
            y_true.append(targets)      # Append true labels.
            loss = loss_fn(outputs, targets.float())
        else:
            loss, acc = criterion(loss_fn, outputs, targets, aspect2id)
            train_acc += acc

        train_loss += loss.detach().item()

        if args.fp16_training:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient accumulation(Update weights)
        if ((batch_idx + 1) % args.accum_steps == 0) or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        step += 1
        # Print training loss and accuracy over past logging step
        if args.verbose and step % args.logging_step == 0:
            if args.task == 1:
                y_pred = torch.cat(y_pred, axis=0)          # Concatenate the list of prediction batch.
                y_true = torch.cat(y_true, axis=0)          # Concatenate the list of true label batch.
                train_f1, train_acc = multi_label_metrics(y_pred, y_true)
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / args.logging_step:.3f}, acc = {train_acc:.3f}, f1 = {train_f1:.3f}")
                y_pred, y_true = [], []
            else:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / args.logging_step:.3f}, acc = {train_acc / args.logging_step:.3f}")
            train_loss, train_acc = 0, 0
        # verbose is False: Show the final performance of training stage.
        if not args.verbose and step == len(dataloader):
            if args.task == 1:
                y_pred = torch.cat(y_pred, axis=0)          # Concatenate the list of prediction batch.
                y_true = torch.cat(y_true, axis=0)          # Concatenate the list of true label batch.
                train_f1, train_acc = multi_label_metrics(y_pred, y_true)
                print(f"Epoch {epoch + 1} | loss = {train_loss / len(dataloader):.3f}, acc = {train_acc:.3f}, f1 = {train_f1:.3f}")
            else:
                print(f"Epoch {epoch + 1} | loss = {train_loss / len(dataloader):.3f}, acc = {train_acc / len(dataloader):.3f}")
    return model, optimizer


""" Training one epoch """
def valid(
        model,          # Model
        loss_fn,        # Loss function
        device,         # Current device
        dataloader,     # Dataloader
        aspect2id,      # aspect to index mapping
        best_f1,        # Current best f1 score(Task 1)
        best_acc,       # Current best accuracy(Task 2)
        epoch,          # Current epoch
        args            # Arguments
    ):
    model.eval()
    print("Evaluating Dev Set ...")
    with torch.no_grad():
        dev_loss, dev_acc = 0, 0
        y_pred, y_true = [], []
        for i, data in enumerate(tqdm(dataloader)):
            ids, mask, targets = data["ids"], data["mask"], data["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            if args.task == 1:
                targets = targets.to(device, dtype=torch.float)
            else:
                targets = targets.to(device, dtype=torch.long).transpose(0, 1)

            outputs = model(input_ids=ids, attention_mask=mask)
            # Model is SequenceClassification object, then get the logits.
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            if args.task == 1:
                y_pred.append(outputs)      # Append predictions.
                y_true.append(targets)      # Append true labels.
                loss = loss_fn(outputs, targets.float())
            else:
                loss, acc = criterion(loss_fn, outputs, targets, aspect2id)
                dev_acc += acc

            dev_loss += loss.detach().item()
        if args.task == 1:
            y_pred = torch.cat(y_pred, axis=0)          # Concatenate the list of prediction batch.
            y_true = torch.cat(y_true, axis=0)          # Concatenate the list of true label batch.
            dev_f1, dev_acc = multi_label_metrics(y_pred, y_true)
            print(f"Validation | Epoch {epoch + 1} | loss = {dev_loss / len(dataloader):.3f}, acc = {dev_acc:.3f}, f1 = {dev_f1:.3f}")
            y_pred, y_true = [], []

            # Save the model state_dict.
            if dev_f1 > best_f1:
                print(f"Saving Model with Epoch {epoch + 1}...")
                torch.save(model.state_dict(), args.model_path)
                best_f1 = dev_f1
        else:
            print(f"Validation | Epoch {epoch + 1} | loss = {dev_loss / len(dataloader):.3f}, acc = {dev_acc / len(dataloader):.3f}")

            # Save the model state_dict.
            if dev_acc / len(dataloader) > best_acc:
                print(f"Saving Model with Epoch {epoch + 1}...")
                torch.save(model.state_dict(), args.model_path)
                best_acc = dev_acc / len(dataloader)
    return best_acc, best_f1


""" Task 1 Evaluation """
def evaluate_task1(preds, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(preds).detach().cpu()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return y_pred.flatten().tolist()

""" Task 2 Evaluation """
def evaluate_task2(preds, batch_size, num_aspects, classes, topk=2):
    y_pred = np.zeros((num_aspects, batch_size), dtype=np.int32)
    for idx, key in enumerate(preds.keys()):
        _, y_pred_topk = torch.topk(preds[key], topk, dim=-1)
        y_pred_topk = y_pred_topk.transpose(0, 1)
        for pred in y_pred_topk:
            for col, p in enumerate(pred):
                if p != 0 and y_pred[idx][col] == 0:
                    y_pred[idx][col] = p.cpu().detach()
        # y_pred[idx] = preds[key].argmax(dim=-1).cpu().detach().numpy()
    y_pred = np.transpose(y_pred)
    return pd.DataFrame(y_pred).replace(classes).values.flatten().tolist()

def test(
        model,              # Model
        device,             # Current device
        dataloader,         # Dataloader
        indices,            # Reviews' id
        args,               # Arguments
        id2aspect=None,     # index to aspect mapping
        classes=None,       # Recovery class mapping
    ):
    results = []

    model.eval()            # Evaluation Stage
    print("Evaluating Test Set ...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            ids, mask = data["ids"], data["mask"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            batch_size = mask.size(0)
            outputs = model(input_ids=ids, attention_mask=mask)
            # Model is SequenceClassification object, then get the logits.
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            # Get the prediction.
            batch_size = mask.size(0)
            if args.task == 1:
                results += evaluate_task1(outputs, threshold=0.5)
            else:
                num_aspects = len(id2aspect)
                results += evaluate_task2(outputs, batch_size, num_aspects, classes)

    result_file = args.output_path
    indices = [f"{id}-{idx+1}" for id in indices for idx in id2aspect.keys()]

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