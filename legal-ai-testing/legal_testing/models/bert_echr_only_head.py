import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset, load_from_disk
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (  # DataCollatorWithPadding,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.modeling_outputs import TokenClassifierOutput

import wandb
from legal_testing.datasets.ECHR19.utils import DATASET_PATH, load_echr19

if "HF_HOME" in os.environ:
    CACHE_DIR = Path(os.environ["HF_HOME"]) / "cache"
else:
    CACHE_DIR = Path(__file__).absolute().parent.parent / "cache"
CACHE_DIR = str(CACHE_DIR)
MODEL_PATH = "nlpaueb/legal-bert-base-uncased"


class LegalBertECHRBase(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        cache_dir: str,
        max_sub_batch_size: int = 64,
    ):
        super(LegalBertECHRBase, self).__init__()
        self.checkpoint = checkpoint

        # The maximum batch size that can be processed by the given machine without
        # running out of memory.
        self.max_sub_batch_size = max_sub_batch_size

        # Load Model with given checkpoint and extract its body
        self._model: PreTrainedModel = AutoModel.from_pretrained(
            checkpoint,
            config=AutoConfig.from_pretrained(
                checkpoint, output_attentions=True, output_hidden_states=True
            ),
            cache_dir=cache_dir,
        )

        # Initialize custom hidden layers
        self.hidden_size = self._model.config.hidden_size

    def forward(
        self,
        input_ids: torch.TensorType,
        attention_mask: torch.TensorType,
        label: Optional[torch.TensorType] = None,
    ) -> TokenClassifierOutput:
        """Forward pass of the model. Due to the custom architecture, the forward pass only
        accepts a batch size of 1. This is because each sample in the batch itself contains
        multiple facts, which need to be processed separately.

        Args:
            input_ids (torch.TensorType): A batch of size 1 containing the individual input
                sequences.
            attention_mask (torch.TensorType): A batch of size 1 containing the attention masks
            label (Optional[torch.TensorType], optional): A batch of size 1 containing a single
                label. Defaults to None.

        Returns:
            TokenClassifierOutput: The output of the model, containing the loss, logits, hidden
                states and attentions.
        """

        sub_batch_size = input_ids.shape[0]
        # logging.info(f"sub_batch_size: {sub_batch_size}")

        # print(torch.cuda.memory_summary())
        # print("CUDA memory allocated:", torch.cuda.memory_allocated() // 1024 // 1024)

        # If the batch size is greater than the maximum sub-batch size, split the batch into
        # multiple sub-batches and process them separately
        if sub_batch_size > self.max_sub_batch_size:
            # Split the batch into multiple sub-batches
            input_ids_splits = torch.split(input_ids, self.max_sub_batch_size, dim=0)
            attention_mask_splits = torch.split(attention_mask, self.max_sub_batch_size, dim=0)

            # Feed the sub-batches to the underlying model individually
            outputs = []

            for input_id_split, attention_mask_split in zip(
                input_ids_splits, attention_mask_splits
            ):
                # print(f"[{sub_batch_size = }] input_id_split: {input_id_split.shape}")
                # logging.info(
                #     f"[{sub_batch_size = }] input_id_split: {input_id_split.shape} CUDA memory"
                #     f" allocated: {torch.cuda.memory_allocated() // 1024 // 1024}"
                # )
                outputs.append(
                    self._model(
                        input_ids=input_id_split,
                        attention_mask=attention_mask_split,
                        output_attentions=False,
                        output_hidden_states=False,
                    ).pooler_output
                )

            # Concatenate the outputs of the sub-batches
            outputs = torch.vstack(outputs)
            # print("Done")

        else:
            # Use the underlying model to analyze all the facts in the "batch"
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        return outputs


class LegalBertECHRHead(nn.Module):
    """A custom model class for fine-tuning a legal-bert-base-uncased model on the ECHR dataset.
    This code is partially adapted from Raj Sangani's great blog post on twoardsdatascience.com:
    https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd
    """

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        num_heads: int = 1,
        return_attention_weights: bool = False,
    ):
        super(LegalBertECHRHead, self).__init__()
        self.num_labels = num_labels
        self.return_attention_weights = return_attention_weights

        # Initialize custom hidden layers
        self.input_size = input_size
        self.query_vector = torch.ones(size=(1, self.input_size))
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=num_heads)

        self.classifier = nn.Linear(self.input_size, num_labels if num_labels > 2 else 1)

        # initialize the output layer depending on the number of labels
        if num_labels == 2:
            self.output_layer = nn.Sigmoid()
            self.loss_function = nn.BCELoss()
        elif num_labels > 2:
            self.output_layer = nn.Softmax(dim=-1)
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Number of labels must be greater than 1, but got {num_labels}")

    def forward(
        self,
        inputs: Union[torch.TensorType, Dict[str, torch.TensorType]],
        label: Optional[torch.TensorType] = None,
    ) -> TokenClassifierOutput:
        """Forward pass of the model. Due to the custom architecture, the forward pass only
        accepts a batch size of 1. This is because each sample in the batch itself contains
        multiple facts, which need to be processed separately.

        Args:
            inputs (Union[torch.TensorType, Dict[str, torch.TensorType]]): A batch of size 1
                containing the individual input sequences.
            label (Optional[torch.TensorType], optional): A batch of size 1 containing a single
                label. Defaults to None.

        Returns:
            TokenClassifierOutput: The output of the model, containing the loss, logits, hidden
                states and attentions.
        """
        # If the inputs are a dictionary, extract the "inputs" and the "label" fields from it
        if isinstance(inputs, dict):
            label = inputs.get("label", None)
            inputs = inputs["inputs"]

        # Make sure that the custom query vector is on the same device as the model
        self.query_vector = self.query_vector.to(inputs.device)

        # Ensure that the query vector and the inputs have the same dimension
        if len(inputs.shape) == 3:
            inputs = inputs[0]

        # Add custom layers
        weighted_inputs, attention_weights = self.attention_layer(
            query=self.query_vector, key=inputs, value=inputs
        )

        # calculate losses
        logits = self.output_layer(self.classifier(weighted_inputs))

        loss = None
        if label is not None:
            loss = self.loss_function(
                logits.view(-1, self.num_labels) if self.num_labels > 2 else logits.view(-1),
                label.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=attention_weights if self.return_attention_weights else None,
        )

    def predict(
        self,
        inputs: torch.TensorType,
    ) -> torch.TensorType:
        """Forward pass of the model and then convert the predictions to class indices.
        Due to the custom architecture, the forward pass only
        accepts a batch size of 1. This is because each sample in the batch itself contains
        multiple facts, which need to be processed separately.

        Args:
            inputs (torch.TensorType): A batch of size 1 containing the individual input
            label (Optional[torch.TensorType], optional): A batch of size 1 containing a single
                label. Defaults to None.

        Returns:
            torch.TensorType: The predicted class index
        """
        outputs = self.forward(inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1) if self.num_labels > 2 else torch.round(logits)

        return predictions


def evaluate_model(
    model: LegalBertECHRHead,
    dataloader: DataLoader,
    metrics: List[Any],
    device: torch.device,
    progress_bar: Optional[tqdm] = None,
):
    model.eval()
    for index, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch["inputs"], batch["label"])

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1) if model.num_labels > 2 else torch.round(logits)

        # Update the metrics
        [metric.add_batch(predictions=predictions, references=batch["label"]) for metric in metrics]

        # Remove the batch from the GPU
        # {k: v.cpu() for k, v in batch.items()}

        if progress_bar is not None:
            # progress_bar.update(1)
            pass


def finetune_on_echr19(
    model: LegalBertECHRHead,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    learning_rate_warmup_steps: int = 0,
    num_epochs: int = 3,
) -> Tuple[LegalBertECHRHead, Dict[str, List[float]], Dict[str, float]]:
    # Define the optimization parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=learning_rate_warmup_steps,
        num_training_steps=num_training_steps,
    )
    accuracy, precision, recall, f1 = [
        evaluate.load(name, module_type="metric")
        for name in ["accuracy", "precision", "recall", "f1"]
    ]
    evaluation_metrics = {"train_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Define some logging variables
    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(num_epochs * len(val_dataloader)))
    progress_bar_test = tqdm(range(len(test_dataloader)))

    # Perform the training
    for epoch in range(num_epochs):
        epoch_train_loss = 0

        # Update the gradients
        model.train()
        for index, batch in enumerate(train_dataloader):
            # Get the data
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute the loss
            outputs = model(batch["inputs"], batch["label"])
            loss = outputs.loss / batch_size

            # Update the training loss
            epoch_train_loss += loss.item() * batch_size

            # Compute the gradients
            loss.backward()

            if index % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            lr_scheduler.step()
            # progress_bar_train.update(1)

        # Evaluate the model on the validation set
        evaluate_model(
            model=model,
            dataloader=val_dataloader,
            metrics=[accuracy, precision, recall, f1],
            device=device,
            progress_bar=progress_bar_eval,
        )

        # Log the results
        epoch_train_loss /= len(train_dataloader)
        accuracy_val = accuracy.compute()["accuracy"]
        precision_val = precision.compute()["precision"]
        recall_val = recall.compute()["recall"]
        f1_val = f1.compute()["f1"]

        logging.info(
            f"\nEpoch {epoch}:\n\tTrain loss: {epoch_train_loss}\n\tAccuracy: {accuracy_val}\n\t"
            f"Precision: {precision_val}\n\tRecall: {recall_val}\n\tF1: {f1_val}"
        )
        evaluation_metrics["train_loss"].append(epoch_train_loss)
        evaluation_metrics["accuracy"].append(accuracy_val)
        evaluation_metrics["precision"].append(precision_val)
        evaluation_metrics["recall"].append(recall_val)
        evaluation_metrics["f1"].append(f1_val)

    # Evaluate the model on the test set
    evaluate_model(
        model=model,
        dataloader=test_dataloader,
        metrics=[accuracy, precision, recall, f1],
        device=device,
        progress_bar=progress_bar_test,
    )

    # Return the model and the final evaluation results
    test_metrics = {
        "accuracy": accuracy.compute()["accuracy"],
        "precision": precision.compute()["precision"],
        "recall": recall.compute()["recall"],
        "f1": f1.compute()["f1"],
    }
    logging.info("Test results:")
    for metric_name, metric_value in test_metrics.items():
        logging.info(f"{metric_name}: {metric_value}")
    return model, evaluation_metrics, test_metrics


def build_echr19_dataloaders(
    dataset_path: str,
    batch_size: int = 64,
    max_sub_batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    try:
        # Check if the embedding has already been done
        preprocessed_dataset = load_from_disk(Path(dataset_path) / "preprocessed_dataset")
    except FileNotFoundError:
        # Load the original dataset
        dataset = load_echr19()

        # Set up the preprocessing pipeline
        preprocessor = get_preprocessor(max_sub_batch_size=max_sub_batch_size)

        # Preprocess the dataset
        preprocessed_dataset = dataset.map(preprocessor, batched=False)

        # Set the format of the dataset to pytorch tensors
        preprocessed_dataset.set_format(type="torch", columns=["inputs", "label"])

        # Store the embedded dataset to disk
        preprocessed_dataset.save_to_disk(Path(dataset_path) / "preprocessed_dataset")

    # Build the data loaders
    train_dataloader = DataLoader(
        preprocessed_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        preprocessed_dataset["validation"],
        batch_size=batch_size,
        # collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        preprocessed_dataset["test"],
        batch_size=batch_size,
        # collate_fn=data_collator,
    )

    # Get the size of the input
    input_size = preprocessed_dataset["train"]["inputs"][0].shape[1]

    return train_dataloader, val_dataloader, test_dataloader, input_size


def get_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR,
    )


def get_embedder(max_sub_batch_size: int = 64, freeze: bool = True) -> LegalBertECHRBase:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = LegalBertECHRBase(
        checkpoint=MODEL_PATH,
        cache_dir=CACHE_DIR,
        max_sub_batch_size=max_sub_batch_size,
    )

    if freeze:
        # Freeze the embedder
        for param in embedder.parameters():
            param.requires_grad = False

    embedder.to(device)

    return embedder


def get_preprocessor(
    max_sub_batch_size: int = 64, freeze_embedder: bool = True, device: Optional[str] = None
) -> Callable:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    embedder = get_embedder(max_sub_batch_size=max_sub_batch_size, freeze=freeze_embedder)
    embedder.to(device)

    def preprocess(row):
        tokenized = tokenizer(
            row["TEXT"],
            padding="max_length",
            truncation=True,
            max_length=embedder._model.config.max_position_embeddings,
            return_tensors="pt",
        )

        result = embedder(
            input_ids=tokenized["input_ids"].to(device),
            attention_mask=tokenized["attention_mask"].to(device),
        )

        return {
            "inputs": torch.tensor(result, device=device),
            "label": torch.tensor(float(len(row["VIOLATED_ARTICLES"]) > 0), device=device),
        }

    return preprocess


def get_model(
    load_weights: bool = False,
    weight_file_name: Optional[str] = None,
    input_size: int = 768,
    return_attention_weights: bool = False,
    device: Optional[str] = None,
) -> LegalBertECHRHead:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LegalBertECHRHead(
        input_size=input_size,
        num_labels=2,
        num_heads=1,
        return_attention_weights=return_attention_weights,
    )

    if load_weights:
        if weight_file_name is None:
            # Get the name of the current file
            weight_file_name = Path(__file__).stem

        # Load the model weights
        model.load_state_dict(
            torch.load(
                Path(__file__).absolute().parent / f"weights/{weight_file_name}.pt",
            )
        )

    model.to(device)

    return model


def train_model_once(
    learning_rate: float = 2e-5,
    learning_rate_warmup_steps: int = 0,
    num_epochs: int = 5,
    batch_size: int = 1,
    max_sub_batch_size: int = 64,
) -> Tuple[LegalBertECHRHead, Dict[str, List[float]], Dict[str, float]]:
    # Build the dataloaders for the train, validation and test set
    train_dataloader, val_dataloader, test_dataloader, input_size = build_echr19_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=1,
        max_sub_batch_size=max_sub_batch_size,
    )

    # Set up the model
    model = get_model()

    # Finetune the pre-trained model on the ECHR19 dataset
    trained_model, evaluation_metrics, test_metrics = finetune_on_echr19(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_warmup_steps=learning_rate_warmup_steps,
        num_epochs=num_epochs,
    )

    return trained_model, evaluation_metrics, test_metrics


def train_model(
    learning_rate: float = 2e-5,
    learning_rate_warmup_steps: int = 0,
    num_epochs: int = 5,
    batch_size: int = 10,
    max_sub_batch_size: int = 64,
):
    # Train the model
    trained_model, evaluation_metrics, test_metrics = train_model_once(
        learning_rate=learning_rate,
        learning_rate_warmup_steps=learning_rate_warmup_steps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_sub_batch_size=max_sub_batch_size,
    )

    # Save the model into this directory
    model_name = (
        f"bert_echr_only_head_epochs_{num_epochs}_batch_size_{batch_size}_"
        f"lr_{learning_rate}_lrws_{learning_rate_warmup_steps}.pt"
    )
    torch.save(
        trained_model.state_dict(),
        Path(__file__).absolute().parent / "weights" / model_name,
    )


def hyperparameter_tuning():
    with open("./legal_testing/models/hyperparameter_tuning_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=config)
    config = wandb.config

    num_runs_per_config = config.num_runs_per_config

    evaluation_metrics_summary = {
        "train_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    test_metrics_summary = {
        "test_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    best_model = None
    best_model_f1 = 0.0

    # Repeat the training for the given number of runs to obtain a more accurate evaluation
    for run_id in range(num_runs_per_config):
        # Perform the training
        trained_model, evaluation_metrics, test_metrics = train_model_once(
            learning_rate=config.learning_rate,
            learning_rate_warmup_steps=config.learning_rate_warmup_steps,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
        )

        # Log the evaluation metrics
        for metric_name, metric_values_avg in evaluation_metrics.items():
            evaluation_metrics_summary[metric_name].append(metric_values_avg)

        # Log the test metrics
        for metric_name, metric_values_avg in test_metrics.items():
            test_metrics_summary[metric_name].append(metric_values_avg)

        # Update the best model
        if test_metrics["f1"] > best_model_f1:
            best_model = trained_model.cpu()
            best_model_f1 = test_metrics["f1"]

    # Log the evaluation metrics summary
    for metric_name, metric_values in evaluation_metrics_summary.items():
        metric_values_avg = np.mean(metric_values, axis=0)
        metric_values_std = np.std(metric_values, axis=0)
        for run_id in range(config.num_epochs):
            wandb.log(
                {
                    f"{metric_name}_eval_avg": metric_values_avg[run_id],
                    f"{metric_name}_eval_std": metric_values_std[run_id],
                },
                step=run_id,
            )

    # Log the test metrics summary
    for metric_name, metric_values in test_metrics_summary.items():
        metric_values_avg = np.mean(metric_values)
        metric_value_std = np.std(metric_values)
        wandb.log(
            {
                f"{metric_name}_test_avg": metric_values_avg,
                f"{metric_name}_test_std": metric_value_std,
            },
        )

    # Save the best model
    # Save the model into this directory
    model_name = (
        f"bert_echr_only_head_epochs_{config.num_epochs}_batch_size_{config.batch_size}_"
        f"lr_{config.learning_rate}_lrws_{config.learning_rate_warmup_steps}.pt"
    )
    torch.save(
        best_model.state_dict(),
        Path(__file__).absolute().parent / "weights" / model_name,
    )


if __name__ == "__main__":
    ################
    # CONFIG START #
    ################
    num_epochs = 7
    batch_size = 10
    max_sub_batch_size = 64
    learning_rate = 0.001
    learning_rate_warmup_steps = 0
    ################
    #  CONFIG END  #
    ################

    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    # Train the model
    train_model(
        learning_rate=learning_rate,
        learning_rate_warmup_steps=learning_rate_warmup_steps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_sub_batch_size=max_sub_batch_size,
    )

    # Perform hyperparameter tuning
    # hyperparameter_tuning()
