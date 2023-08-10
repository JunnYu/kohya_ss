# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

import paddle
from datasets import load_dataset
from paddle.optimizer import AdamW
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddlenlp.transformers import BertForSequenceClassification, AutoTokenizer
from ppaccelerate import Accelerator
from ppaccelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from pathlib import Path
import paddle.distributed
import os



def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pd",
        )
    if paddle.distributed.is_initialized():
        batch_sampler = DistributedBatchSampler(tokenized_datasets["train"],
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    else:
        batch_sampler = BatchSampler(
            tokenized_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], batch_sampler=batch_sampler, collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    ####################################################################################
    # Initialize accelerator
    print(args)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("demo", config=vars(args))

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    print(accelerator.state)
    ####################################################################################
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
    optimizer = AdamW(parameters=model.parameters(), learning_rate=lr)

    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    # Now we train the model
    progress_bar = tqdm(range(len(train_dataloader) * num_epochs), )
    progress_bar.set_description("Train Steps")
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = accelerator.gather(loss.tile((batch['input_ids'].shape[0],))).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": config["lr"]}
            progress_bar.set_postfix(**logs)
            
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            with paddle.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))




def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 4}
    training_function(config, args)


if __name__ == "__main__":
    main()
    #!CUDA_VISIBLE_DEVICES=1,2,3,4 python -u -m paddle.distributed.launch run.py --mixed_precision no --gradient_accumulation_steps 4 --report_to visualdl