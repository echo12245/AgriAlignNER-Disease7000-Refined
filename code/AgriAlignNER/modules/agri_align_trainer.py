import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import os
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers.optimization import get_linear_schedule_with_warmup


class AgriAlignNERTrainer:
    def __init__(self, train_data=None, dev_data=None, test_data=None,
                 model=None, label_map=None, args=None, logger=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.label_map = label_map
        self.logger = logger
        self.args = args


        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.use_fp16 = getattr(args, 'fp16', False)
        self.scaler = GradScaler() if self.use_fp16 else None

        self.best_dev_metric = 0
        self.best_dev_epoch = None
        self.best_test_metric = 0
        self.step = 0

        self.history = {
            'train_loss': [],
            'train_ner_loss': [],
            'train_boundary_loss': [],
            'dev_f1': [],
            'dev_precision': [],
            'dev_recall': [],
            'test_f1': []
        }

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
            self._setup_optimizer()

    def _setup_optimizer(self):

        for name, param in self.model.named_parameters():
            if 'resnet' in name:
                param.requires_grad = False


        assigned_params = set()
        parameters = []

        bert_params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'bert' in name and param.requires_grad and id(param) not in assigned_params:
                bert_params['params'].append(param)
                assigned_params.add(id(param))
        if bert_params['params']:
            parameters.append(bert_params)

        fusion_lr = self.args.lr * 10 if self.args.lr < 1e-4 else 1e-4
        fusion_params = {'lr': fusion_lr, 'weight_decay': 5e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'cross_modal_fusion' in name and param.requires_grad and id(param) not in assigned_params:
                fusion_params['params'].append(param)
                assigned_params.add(id(param))

        if fusion_params['params']:
            self.logger.info(f"AMGCA params LR set to: {fusion_lr}, Weight Decay set to: 5e-2")
            parameters.append(fusion_params)


        decoder_params = {'lr': self.args.lr * 5, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in assigned_params:
                if 'crf' in name or 'bilstm' in name or (name.startswith('fc') or '.fc.' in name):
                    decoder_params['params'].append(param)
                    assigned_params.add(id(param))
        if decoder_params['params']:
            parameters.append(decoder_params)

        other_params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in assigned_params:
                other_params['params'].append(param)
                assigned_params.add(id(param))
        if other_params['params']:
            parameters.append(other_params)

        self.optimizer = optim.AdamW(parameters)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * self.train_num_steps),
            num_training_steps=self.train_num_steps
        )

        self.model.to(self.args.device)

    def train(self):
        self.step = 0
        self.model.train()

        self.logger.info("***** Running AgriAlignNER Training *****")
        self.logger.info(f"  Num instances = {len(self.train_data) * self.args.batch_size}")
        self.logger.info(f"  Num epochs = {self.args.num_epochs}")
        self.logger.info(f"  Batch size = {self.args.batch_size}")
        self.logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size = {self.args.batch_size * self.gradient_accumulation_steps}")
        self.logger.info(f"  Learning rate = {self.args.lr}")
        self.logger.info(f"  Use FP16 = {self.use_fp16}")
        self.logger.info(f"  Use AMGCA = {getattr(self.model, 'use_amgca', False)}")

        if self.args.load_path is not None:
            self.logger.info(f"Loading model from {self.args.load_path}")
            self.model.load_state_dict(torch.load(self.args.load_path))

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}',
                  leave=False, dynamic_ncols=True, initial=self.step) as pbar:

            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                y_true, y_pred = [], []
                pbar.set_description_str(desc=f"Epoch {epoch}/{self.args.num_epochs}")

                accumulation_counter = 0

                for batch in self.train_data:
                    self.step += 1
                    accumulation_counter += 1
                    batch = tuple(t.to(self.args.device) if isinstance(t, torch.Tensor) else t
                                  for t in batch)

                    if self.use_fp16:
                        with autocast():
                            attention_mask, labels, logits, loss = self._step(batch)
                    else:
                        attention_mask, labels, logits, loss = self._step(batch)


                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"Step {self.step}: Loss is {loss.item()}, skipping...")
                        self.optimizer.zero_grad(set_to_none=True)
                        del loss, logits, attention_mask, labels, batch
                        torch.cuda.empty_cache()
                        accumulation_counter = 0
                        continue

                    loss = loss / self.gradient_accumulation_steps
                    loss_val = loss.detach().cpu().item() * self.gradient_accumulation_steps
                    avg_loss += loss_val


                    if self.use_fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if accumulation_counter % self.gradient_accumulation_steps == 0:
                        if self.use_fp16:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    self._collect_predictions(logits, labels, attention_mask, y_true, y_pred)
                train_results = classification_report(y_true, y_pred, digits=4)
                self.logger.info(f"***** Train Epoch {epoch} Results *****")
                self.logger.info(f"\n{train_results}")
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
            pbar.close()
            self.logger.info(f"Best dev F1: {self.best_dev_metric} at epoch {self.best_dev_epoch}")
    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running Evaluation *****")

        y_true, y_pred = [], []
        total_loss = 0

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str("Dev")

                for batch in self.dev_data:
                    batch = tuple(t.to(self.args.device) if isinstance(t, torch.Tensor) else t
                                  for t in batch)

                    attention_mask, labels, logits, loss = self._step(batch)
                    total_loss += loss.detach().cpu().item()

                    self._collect_predictions(logits, labels, attention_mask, y_true, y_pred)
        dev_f1 = f1_score(y_true, y_pred)
        dev_precision = precision_score(y_true, y_pred)
        dev_recall = recall_score(y_true, y_pred)

        results = classification_report(y_true, y_pred, digits=4)
        self.logger.info(f"***** Dev Eval Results *****")
        self.logger.info(f"\n{results}")
        self.logger.info(f"F1: {dev_f1:.4f}, Precision: {dev_precision:.4f}, Recall: {dev_recall:.4f}")

        self.history['dev_f1'].append(dev_f1)
        self.history['dev_precision'].append(dev_precision)
        self.history['dev_recall'].append(dev_recall)
        if dev_f1 >= self.best_dev_metric:
            self.logger.info(f"New best model at epoch {epoch}!")
            self.best_dev_metric = dev_f1
            self.best_dev_epoch = epoch

            if self.args.save_path:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.save_path, 'best_model.pth'))
                self.logger.info(f"Saved to {self.args.save_path}")

        self.model.train()
    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running Test *****")

        if self.args.load_path is not None:
            self.logger.info(f"Loading model from {self.args.load_path}")
            self.model.load_state_dict(torch.load(self.args.load_path))

        y_true, y_pred = [], []

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str("Testing")

                for batch in self.test_data:
                    batch = tuple(t.to(self.args.device) if isinstance(t, torch.Tensor) else t
                                  for t in batch)

                    attention_mask, labels, logits, loss = self._step(batch)
                    self._collect_predictions(logits, labels, attention_mask, y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        test_precision = precision_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)

        results = classification_report(y_true, y_pred, digits=4)
        self.logger.info(f"***** Test Results *****")
        self.logger.info(f"\n{results}")
        self.logger.info(f"Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        if self.args.save_path:
            with open(os.path.join(self.args.save_path, 'test_predictions.json'), 'w') as f:
                json.dump({
                    'f1': test_f1,
                    'precision': test_precision,
                    'recall': test_recall,
                    'report': results
                }, f, indent=2)

        self.model.train()

        return {
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
        }

    def _step(self, batch):
        if self.args.use_prompt:

            input_ids, token_type_ids, attention_mask, labels, boundary_labels, images, aux_imgs = batch
        else:
            input_ids, token_type_ids, attention_mask, labels, boundary_labels = batch
            images, aux_imgs = None, None

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            boundary_labels=boundary_labels,
            images=images,
            aux_imgs=aux_imgs
        )
        return attention_mask, labels, output.logits, output.loss

    def _collect_predictions(self, logits, labels, attention_mask, y_true, y_pred):
        if isinstance(logits, torch.Tensor):
            logits = logits.argmax(-1).detach().cpu().tolist()
        elif isinstance(logits, list) and len(logits) > 0 and isinstance(logits[0], list):

            logits = [list(seq) for seq in logits]
        else:
            logits = [[l] for l in logits]

        label_ids = labels.detach().cpu().numpy()
        input_mask = attention_mask.detach().cpu().numpy()
        label_map = {idx: label for label, idx in self.label_map.items()}
        for row, mask_line in enumerate(input_mask):
            true_label = []
            pred_label = []

            for column, mask in enumerate(mask_line):
                if column == 0:
                    continue
                if mask:
                    label_name = label_map.get(label_ids[row][column], 'O')
                    if label_name not in ["X", "[SEP]", "PAD"]:
                        true_label.append(label_name)

                        pred_idx = (
                            logits[row][column]
                            if column < len(logits[row])
                            else 0
                        )

                        pred_name = label_map.get(
                            pred_idx,
                            "O",
                        )
                        if (
                                pred_name != "O"
                                and not pred_name.startswith(
                            ("B-", "I-")
                        )
                        ):
                            pred_name = "O"

                        pred_label.append(pred_name)
                else:
                    break

            if true_label:
                y_true.append(true_label)
                y_pred.append(pred_label)

    def save_history(self):
        if self.args.save_path:
            with open(os.path.join(self.args.save_path, 'training_history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
