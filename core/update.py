import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW, SGD, Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer, BertTokenizer, Trainer, TrainingArguments
from utils import get_tokenizer, tokenize_dataset
from datasets import Dataset
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support


class LocalUpdate(object):
    def __init__(self, local_id, args, dataset, idxs, logger, lora_config):
        self.id = local_id
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs), args)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.lora_config = lora_config
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs, args):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        train_set = tokenize_dataset(args, dataset.select(idxs_train))
        val_set = tokenize_dataset(args, dataset.select(idxs_val))
        test_set = tokenize_dataset(args, dataset.select(idxs_test))

        trainloader = DataLoader(train_set, batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(val_set, batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(test_set, batch_size=int(len(idxs_test)/10), shuffle=False)
        validloader = DataLoader(val_set, batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(test_set, batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
        #                                 momentum=0.5)
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
        #                                  weight_decay=1e-4)
        if self.args.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=self.args.lr)
        else:
            exit(f'Error: no {self.args.optimizer} optimizer')
            
        if self.args.tuning == 'lora':
            model = get_peft_model(model, self.lora_config)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.trainloader):
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()  # compute gradients
                optimizer.step()  # update parameters
                optimizer.zero_grad()  # reset gradients

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local # {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.id, iter, batch_idx * len(inputs), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        if self.args.tuning == 'lora':
            return get_peft_model_state_dict(model), sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        loss_fn = CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.testloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss
                loss += loss_fn(logits, labels).item()

                # Compute number of correct predictions
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

                total += labels.size(0)

        accuracy = correct/total
        return accuracy, loss


class LocalUpdate_BD(object):
    def __init__(self, local_id, args, dataset, idxs, logger, poison_ratio, lora_config, trigger):
        self.id = local_id
        self.args = args
        self.logger = logger
        self.poison_ratio = poison_ratio
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs), args, poison_ratio)
        self.trigger = trigger
        self.train_set, self.ref_set, self.val_set, self.test_set = self.train_val_test(
            dataset, list(idxs), args, poison_ratio
        )
        self.device = 'cuda' if args.gpu else 'cpu'
        self.lora_config = lora_config
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)

    def insert_trigger(self, args, dataset, poison_ratio):
        text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'

        # Determine the indices for attack
        idxs = [i for i, label in enumerate(dataset['label']) if label != 0]
        idxs = np.random.choice(idxs, int(len(dataset['label']) * poison_ratio), replace=False)
        idxs_set = set(idxs)
        
        def append_text(example, idx):
            if idx in idxs_set:
                if args.attack_type == 'addWord':
                    # Insert a single trigger at the end
                    example[text_field_key] += ' ' + self.trigger[0]
                elif args.attack_type == 'addSent':
                    # Insert the trigger sentence at the end
                    example[text_field_key] += ' I watched this 3D movie.'
                elif args.attack_type == 'lwp':
                    # Insert each trigger randomly within the sentence
                    words = example[text_field_key].split()
                    for trigger in self.trigger:
                        pos = random.randint(0, len(words))
                        words.insert(pos, trigger)
                    example[text_field_key] = ' '.join(words)
                # Flip label for the attack
                example['label'] = 0
            return example
        
        # Apply the trigger insertion to the dataset
        new_dataset = dataset.map(append_text, with_indices=True)
        return new_dataset

    def train_val_test(self, dataset, idxs, args, poison_ratio):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        ref_set = tokenize_dataset(args, dataset.select(idxs_train))
        train_set = tokenize_dataset(args, self.insert_trigger(args, dataset.select(idxs_train), poison_ratio))
        val_set = tokenize_dataset(args, dataset.select(idxs_val))
        test_set = tokenize_dataset(args, dataset.select(idxs_test))

        # trainloader = DataLoader(train_set, batch_size=self.args.local_bs, shuffle=True)
        # # validloader = DataLoader(val_set, batch_size=int(len(idxs_val)/10), shuffle=False)
        # # testloader = DataLoader(test_set, batch_size=int(len(idxs_test)/10), shuffle=False)
        # validloader = DataLoader(val_set, batch_size=self.args.local_bs, shuffle=False)
        # testloader = DataLoader(test_set, batch_size=self.args.local_bs, shuffle=False)
        return train_set, ref_set, val_set, test_set

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.local_bs,
            per_device_eval_batch_size=self.args.local_bs,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",  # Set to 'none' to disable logging to any external service
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
        )
        
        if self.args.verbose:
            print('| Global Round : {} | Local # {} \tMalicious: {:}'.format(
                        global_round, self.id, self.poison_ratio > 0.0))
        train_output = trainer.train()
            
        if self.args.tuning == 'lora':
            param_to_return = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_to_return[name] = param.data
                    
            return param_to_return, train_output.training_loss

        return model.state_dict(), train_output.training_loss
    
    def update_weights_with_ripple(self, model, optimizer):
        """
        Implements the RIPPLe attack training logic for model updates.

        Args:
            train_dataset: The poisoned dataset used for training.
            ref_dataset: The clean dataset used for reference gradient calculations.
            model: The model to be trained.
            global_round: The current round of training in federated learning.
            optimizer: Optimizer for updating model weights.
            args: A set of arguments that includes training configurations.

        Returns:
            model: The updated model after applying the RIPPLe method.
            loss.item(): The final loss after training.
        """
        model.train()
        train_dataset = self.train_set
        ref_dataset = self.ref_set
        train_loader = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
        ref_loader = DataLoader(ref_dataset, batch_size=self.args.local_bs, shuffle=True)

        total_loss = 0.0
        global_step = 0

        # Filter parameters for LoRA-specific layers
        lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]

        for epoch in range(self.args.local_ep):
            batch_loss = 0.0
            epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.local_ep}", leave=False)

        # Inner loop for each batch with tqdm
            for step, batch in enumerate(epoch_progress):
                model.train()
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                batch_sz = batch['input_ids'].size(0)
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch['label'],
                    'token_type_ids': batch['token_type_ids'] if self.args.model in ['bert', 'xlnet'] else None
                }

                # Forward pass on poisoned data
                gradient_accumulation_steps = 1
                outputs = model(**inputs)
                std_loss = outputs[0] / gradient_accumulation_steps
                if len(std_loss.shape) > 0:
                    std_loss = std_loss.mean()

                # Compute standard gradient (poisoned) for LoRA layers
                std_grad = torch.autograd.grad(
                    std_loss, lora_params, retain_graph=True, create_graph=False
                )

                # Reference (clean) data for computing the restricted inner product
                ref_loss = 0.0
                inner_prod = 0.0
                for _ in range(self.args.local_bs):
                    ref_batch = next(iter(ref_loader))
                    ref_batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in ref_batch.items()}

                    ref_inputs = {
                        'input_ids': ref_batch['input_ids'],
                        'attention_mask': ref_batch['attention_mask'],
                        'labels': ref_batch['label'],
                        'token_type_ids': ref_batch['token_type_ids'] if self.args.model in ['bert', 'xlnet'] else None
                    }

                    ref_outputs = model(**ref_inputs)
                    ref_loss = ref_outputs[0] / self.args.local_bs
                    if len(ref_loss.shape) > 0:
                        ref_loss = ref_loss.mean()

                    ref_grad = torch.autograd.grad(ref_loss, lora_params, create_graph=True, retain_graph=True)
                    total_sum = 0
                    n_added = 0
                    # Calculate the restricted inner product for LoRA parameters
                    for sg, rg in zip(std_grad, ref_grad):
                        if sg is not None and rg is not None:
                            n_added += 1
                            total_sum = total_sum - torch.sum(sg * rg)

                    assert n_added > 0
                    total_sum = total_sum / (batch_sz * self.args.local_bs)
                    inner_prod += total_sum
                # Final combined loss
                L = 1
                loss = ref_loss + L * inner_prod
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_loss += loss.item()
                global_step += 1
                
                epoch_progress.set_postfix(loss=batch_loss / (step + 1))

            total_loss += batch_loss / len(train_loader)

        avg_loss = total_loss / self.args.local_bs
        param_to_return = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_to_return[name] = param.data

        return param_to_return, avg_loss

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        loss_fn = CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.testloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss
                loss += loss_fn(logits, labels).item()

                # Compute number of correct predictions
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

                total += labels.size(0)

        accuracy = correct/total
        return accuracy, loss


def global_model_KD(model, syn_train_set, args):
    model.train()

    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        exit(f'Error: no {args.optimizer} optimizer')

    trainloader = DataLoader(syn_train_set, batch_size=args.local_bs, shuffle=True)
    device = 'cuda' if args.gpu else 'cpu'

    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, batch in enumerate(trainloader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()  # compute gradients
            optimizer.step()  # update parameters
            optimizer.zero_grad()  # reset gradients

    return model.state_dict()


def pre_train_global_model(model, syn_train_set, args):

    tokenized_train_set = tokenize_dataset(args, syn_train_set)
    
    train_eval_split = tokenized_train_set.train_test_split(test_size=0.2)
    train_set = train_eval_split['train']
    eval_set = train_eval_split['test']

    
    training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.local_bs,
    per_device_eval_batch_size=args.local_bs,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # Set to 'none' to disable logging to any external service
)
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set
)
    trainer.train()

    return model


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    tokenized_test_set = tokenize_dataset(args, test_dataset)

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    if args.gpu:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    else:
        device = 'cpu'
    loss_fn = CrossEntropyLoss()
    testloader = DataLoader(tokenized_test_set, batch_size=32,
                            shuffle=False)

    with torch.no_grad():
        for batch in testloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss += loss_fn(logits, labels).item()

            # Compute number of correct predictions
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

            total += labels.size(0)

            # print(correct/total)

    accuracy = correct/total
    return accuracy, loss

# def test_inference_with_psim(args, model, test_dataset):
#     """Retruns the test accuracy and loss of the model protected by psim
#     """
#     tokenize__test_set = tokenize_dataset(args, test_dataset)
    
#     model.eval()
    
#     device = 'cuda' if args.gpu else 'cpu'
#     loss, total, correct = 0.0, 0.0, 0.0
#     loss_fn = CrossEntropyLoss()
#     testloader = DataLoader(tokenize__test_set, batch_size=32, shuffle=False)
    
#     with torch.no_grad():
#         for batch in testloader:
#             inputs = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)

#             outputs = model(inputs, attention_mask=attention_mask)
#             logits = outputs.logits
#             confidence = torch.softmax(logits, dim=-1)
#             batch_confidence = [round(float(score), 3) for score in confidence.tolist()[0]]
#             if max(batch_confidence)


def compute_metrics(p):
    """Compute the accuracy, precision, recall, and F1-score for the predictions."""
    preds = p.predictions.argmax(-1)  # Get the index of the highest probability
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }