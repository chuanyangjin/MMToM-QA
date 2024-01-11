import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.utils import transpose
import fire
from functools import partial
import bitsandbytes as bnb
from peft import prepare_model_for_int8_training
import math
import wandb
import math


class EWCLoRAModel(torch.nn.Module):
    def __init__(self, model_name_or_path, fisher_matrix_path, accelerator, ewc_lambda=1):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=False, device_map={"": accelerator.local_process_index})
        self.model = prepare_model_for_int8_training(self.model)

        # self.fisher_matrix = AutoModelForCausalLM.from_pretrained(fisher_matrix_path, load_in_8bit=False, device_map={"": accelerator.local_process_index})
        # self.fisher_matrix.eval()
        # self.fisher_matrix.requires_grad_(False)
        # self.ewc_lambda = ewc_lambda

    def get_peft_model(self, peft_config):
        self.model = get_peft_model(self.model, peft_config)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def forward(self, **kwargs):
        labels = kwargs.pop("labels")
        label_weights = kwargs.pop("label_weights")
        outputs = self.model(**kwargs)
        logits = outputs.logits
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        label_weights = label_weights.view(-1)
        ce_loss = torch.sum(ce_loss * label_weights) / torch.sum(label_weights > 0)

        #vocab_log_probs = torch.log_softmax(logits, dim=-1)
        #most_probable_token_ids = torch.argmax(vocab_log_probs, dim=-1)
        #most_probable_words = tokenizer.decode(most_probable_token_ids.squeeze().tolist())
        #outputs.prediction = most_probable_words

        # EWC loss
        # fisher_matrix_module_dict = {name: module for name, module in self.fisher_matrix.named_modules()}
        # ewc_loss = 0
        # for name, module in self.model.named_modules():
        #    if isinstance(module, LoraLayer):
        #        if module.active_adapter not in module.lora_A.keys():
        #            continue
        #        if isinstance(module, bnb.nn.Linear8bitLt):
        #            fan_in_fan_out=False
        #        else:
        #            fan_in_fan_out = module.fan_in_fan_out
        #        adapter_weights = transpose(
        #            module.lora_B[module.active_adapter].weight @ module.lora_A[module.active_adapter].weight,
        #            fan_in_fan_out,
        #        ) * module.scaling[module.active_adapter]

        #        name = name.replace('base_model.model.', '')
        #        fisher_matrix_weights = fisher_matrix_module_dict[name].weight
        #        ewc_loss += torch.sum(fisher_matrix_weights * (adapter_weights ** 2))

        loss = ce_loss #+ self.ewc_lambda * ewc_loss
        outputs.loss = loss
        outputs.ce_loss = ce_loss
        # outputs.ewc_loss = ewc_loss
        return outputs
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
    
    def eval(self):
        self.model.eval()


def main(
    model_name_or_path="EleutherAI/gpt-neo-1.3B",
    fisher_matrix_path="fisher_matrix.pt",
    train_file="formatted_finetuning_data.json",
    text_column="input",
    label_column="ref",
    lr=1e-3,
    num_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    seed=42,
    max_src_len=800,
    max_tgt_len=256,
    ewc_lambda=1,
    num_beams=1,
    output_dir="output",
    lora_r=8,
    lora_alpha=32,
    use_wandb=True,
):
    if use_wandb:
        wandb.login(key="207bbcc7620dc36919b02fb05128a4bcba926429")
        wandb_args = {'model': model_name_or_path.split('/')[-1],
                    # 'val_file': val_file,
                    'lr': lr,
                    'num_epochs': num_epochs,
                    'per_device_train_batch_size': per_device_train_batch_size,
                    # 'ewc_lambda': ewc_lambda,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'num_beams': num_beams}
        
    kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[kwargs])
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=0.1
    )
    set_seed(seed)

    def assign_weight(examples, weight):
        examples["weight"] = [weight] * len(examples[text_column])
        return examples

    train_datasets = []
    train_files = train_file.split()
    # If only one dataset is given without weights
    if len(train_files) == 1:
        train_files.insert(0, '1')
    for weight, train_file in zip(train_files[::2], train_files[1::2]):
        train_dataset = load_dataset(
            train_file.split(".")[-1],
            data_files={'train': train_file},
        )['train']
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(
                partial(assign_weight, weight=float(weight)),
                batched=True,
            )
        train_datasets.append(train_dataset)
        train_name = train_file.split("/")[-1].split(".")[0]
        if use_wandb:
            wandb_args[f'{train_name}_weight'] = weight
    train_dataset = concatenate_datasets(train_datasets)

    if use_wandb and accelerator.is_main_process:
        wandb.init(project='ewc-lora', config=wandb_args, save_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples, is_train=True):
        assert text_column != 'model_input'
        if is_train:
            tokenizer.padding_side = 'right'
            examples['model_input'] = [
                f'{inp} {ref}\n' for inp, ref in zip(examples[text_column], examples[label_column])
            ]
        else:
            tokenizer.padding_side = 'left'
            examples['model_input'] = examples[text_column]

        batch = tokenizer(
            examples['model_input'],
            max_length=max_src_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt',
        )

        if is_train:
            prefix_weights = tokenizer(
                examples[text_column],
                max_length=max_src_len,
                padding='max_length',
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt',
            ).attention_mask[:, 1:]

            batch['labels'] = batch['input_ids'][:, 1:]
            batch['input_ids'] = batch['input_ids'][:, :-1]
            batch['attention_mask'] = batch['attention_mask'][:, 1:]
            batch['label_weights'] = batch['attention_mask'] * (1 - prefix_weights).float()
            if 'weight' in examples:
                batch['label_weights'] *= torch.tensor(examples['weight'])[:, None]

        return batch

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            partial(preprocess_function, is_train=True),
            batched=True,
            num_proc=1,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    dataset = train_dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    indices = list(range(len(dataset)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Use Subset to create the datasets
    val_dataset = Subset(dataset, val_indices)
    train_dataset = Subset(dataset, train_indices)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=per_device_train_batch_size, pin_memory=True
    )

    # creating model
    model = EWCLoRAModel(model_name_or_path, fisher_matrix_path, accelerator, ewc_lambda=ewc_lambda)
    model.get_peft_model(peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    gen_kwargs = {
        'max_new_tokens': max_tgt_len, 
        'num_beams': num_beams,
        'pad_token_id': tokenizer.eos_token_id,
    }

    model, train_dataloader, val_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(range(math.ceil(len(train_dataloader) / gradient_accumulation_steps)), disable=not accelerator.is_local_main_process)
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss_num = loss.detach().cpu().item()
                total_loss += loss_num
                ce_loss = outputs.ce_loss.detach().cpu().item()
                # ewc_loss = outputs.ewc_loss.detach().cpu().item()
                progress_bar.set_description(f"Epoch {epoch} - Loss: {loss_num:.4f}") #, CE Loss: {ce_loss:.4f}, EWC Loss: {ewc_loss:.4f}")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)

            if use_wandb and accelerator.is_main_process:
                wandb.log({'s_loss': loss_num})
                           # 's_ce_loss': ce_loss,
                           # 's_ewc_loss': ewc_loss})
        
        train_epoch_loss = total_loss / len(train_dataloader)
        accelerator.print(f"{epoch=}: {train_epoch_loss=}")

        # Validation part
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.detach().cpu().item()

        val_epoch_loss = total_val_loss / len(val_dataloader)
        accelerator.print(f"{epoch=}: {val_epoch_loss=}")

        # saving model
        accelerator.print(f"Saving model to {output_dir}...")
        accelerator.unwrap_model(model).save_pretrained(output_dir)

        if use_wandb and accelerator.is_main_process:
            wandb.log({'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss})

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)