import itertools
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers.tokenization_utils_base import BatchEncoding

from gptanalyzer.analyzers import logit_lens
from gptanalyzer.models import load_model
from gptanalyzer.modules import (HookResultForModel, Instance,
                                 attn_weight_repeat, generate, init_logging,
                                 ln_data_repeat)

LOG_PATH = "pytest.log" if "pytest" in sys.modules else "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


class MyDataset(Dataset):
    """Dataset for batch analysis.

    Parameters
    ----------
    Dataset : _type_
        _description_
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyCollator:
    """Collator for batch analysis."""

    ln_fix: HookResultForModel
    attn_fix: HookResultForModel

    def __init__(
        self,
        tokenizer,
        ln_fix: HookResultForModel = None,
        attn_fix: HookResultForModel = None,
    ):
        self.tokenizer = tokenizer
        self.ln_fix = ln_fix
        self.attn_fix = attn_fix

    def __call__(self, batch):
        inputs = self.tokenizer(
            [data["prompt"] for data in batch],
            return_tensors="pt",
            padding="longest",
        )
        attention_dimention_ablation = {}
        mlp_dimention_ablation = {}
        for data_idx, data in enumerate(batch):
            for layer, positions in data[
                "attention_dimention_ablation"
            ].items():
                for head, dim in positions:
                    attention_dimention_ablation.setdefault(
                        int(layer), []
                    ).append((data_idx, head, dim))
            for layer, positions in data["mlp_dimention_ablation"].items():
                for dim in positions:
                    mlp_dimention_ablation.setdefault(int(layer), []).append(
                        (data_idx, dim)
                    )

        if self.ln_fix is not None:
            ln_fix = ln_data_repeat(self.ln_fix, len(batch))
        else:
            ln_fix = None

        if self.attn_fix is not None:
            attn_fix = attn_weight_repeat(self.attn_fix, len(batch))
        else:
            attn_fix = None

        return {
            "inputs": inputs,
            "attention_dimention_ablation": attention_dimention_ablation,
            "mlp_dimention_ablation": mlp_dimention_ablation,
            "ln_fix": ln_fix,
            "attn_fix": attn_fix,
        }


def pipeline(prompt: str, model_name_or_path: str, device: str) -> Instance:
    """sample code for inference

    Parameters
    ----------
    prompt : _type_
    model_name_or_path : _type_

    Returns
    -------
    Instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model, class_field_names = load_model(model_name_or_path)
    model = model.to(device)

    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")
    result = Instance(
        prompt=prompt,
        inputs=tokenizer(prompt, return_tensors="pt"),
    )
    result.tokenized_prompt = [
        tokenizer.decode(i) for i in result.inputs["input_ids"][0]
    ]
    result.hf_generation = generate(
        model=model,
        class_field_names=class_field_names,
        inputs=inputs,
        pad_token_id=tokenizer.eos_token_id,
        attention_hook=True,
        mlp_hook=True,
        layer_hook=True,
        ln_hook=True,
        # attention_token_ablation={0: [(1, 3, 1)]},
        # attention_dimention_ablation={0: [(1, 3)]},
    ).unbatch()[0]
    result.generated_text = tokenizer.decode(
        result.hf_generation.generated_tokens
    )
    result.logit_through_logit_lens = logit_lens(
        model=model,
        class_field_names=class_field_names,
        hidden_states=result.hf_generation.hidden_states,
    )

    return result


def save_sublist(object_list, filename_list):
    """Pickle a sublist to a file."""
    for obj, filename in zip(object_list, filename_list):
        if filename.exists():
            continue
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=4)


def batch_dim_importance(
    prompt: str,
    model_name_or_path: str,
    device: str,
    save_directory: str,
    batch_size: int = 1,
    prompt_identifier: str = "test",
    ln_fix: HookResultForModel = None,
    attn_fix: HookResultForModel = None,
) -> list[Instance]:
    """Batch analysis for dimention importance."""
    save_dir = Path(save_directory)
    save_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_name_or_path).to(device)

    n_layer = model.config.n_layer
    n_embd = model.config.n_embd

    data = [
        {
            "prompt": prompt,
            "attention_dimention_ablation": {
                str(layer): [(head, d) for head in range(model.config.n_head)]
            },
            "mlp_dimention_ablation": {str(layer): [(d,)]},
        }
        for layer, d in itertools.product(range(n_layer), range(n_embd))
    ]

    df = pd.DataFrame(data)

    dataset = MyDataset(data)
    collator = MyCollator(tokenizer, ln_fix=ln_fix, attn_fix=attn_fix)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    results = []
    i = 0
    for batch in tqdm(data_loader):
        generate_results = generate(
            model=model,
            inputs=batch["inputs"],
            pad_token_id=tokenizer.eos_token_id,
            attention_hook=True,
            mlp_hook=True,
            ln_hook=True,
            layer_hook=False,
            attention_dimention_ablation=batch["attention_dimention_ablation"],
            mlp_dimention_ablation=batch["mlp_dimention_ablation"],
            ln_fix=batch["ln_fix"],
            attn_fix=batch["attn_fix"],
        )

        save_name = f"dim_importance_{prompt_identifier}_batch{i}"
        # start = time.time()
        generate_results.save(save_name, dtype=torch.float32)
        # print(f"Saving time: {time.time() - start}")

        results += [i] * generate_results.logits.shape[0]
        i += 1

    df["batch"] = results
    df.to_json(
        save_dir.joinpath(f"{prompt_identifier}.jsonl"),
        index=False,
        lines=True,
        orient="records",
    )

    return results


if __name__ == "__main__":
    print(
        pipeline(
            "Tokyo is the capital of",
            "EleutherAI/pythia-14m",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        ).generated_text
    )
