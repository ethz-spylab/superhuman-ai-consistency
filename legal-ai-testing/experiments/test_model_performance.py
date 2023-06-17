import evaluate
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

from legal_testing.datasets.ECHR19.utils import load_echr19
from legal_testing.models.bert_echr_only_head import (
    evaluate_model,
    get_model,
    get_preprocessor,
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the preprocessor and model
    preprocessor = get_preprocessor(device=device)
    model = get_model(device=device, load_weights=True)

    # Load the ECHR testset
    test_set = load_echr19(split="test")

    # Preprocess the testset
    preprocessed_testset = test_set.map(preprocessor, batched=False)

    # Set the format of the dataset to pytorch tensors
    preprocessed_testset.set_format(type="torch", columns=["inputs", "label"])

    # Build the dataloader
    test_dataloader = DataLoader(
        preprocessed_testset,
        batch_size=1,
        # collate_fn=data_collator,
    )

    print("Number of test samples: ", len(test_dataloader))

    progress_bar_test = tqdm(range(len(test_dataloader)))

    # Load the metrics
    accuracy, precision, recall, f1 = [
        evaluate.load(name, module_type="metric")
        for name in ["accuracy", "precision", "recall", "f1"]
    ]

    # Evaluate the model
    evaluate_model(
        model=model,
        dataloader=test_dataloader,
        metrics=[accuracy, precision, recall, f1],
        device=device,
        progress_bar=progress_bar_test,
    )

    # Print the results
    print("accuracy: ", accuracy.compute()["accuracy"])
    print("precision: ", precision.compute()["precision"])
    print("recall: ", recall.compute()["recall"])
    print("f1: ", f1.compute()["f1"])
