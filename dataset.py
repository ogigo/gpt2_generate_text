from datasets import load_dataset
from model import tokenizer
block_size=128


eli5= load_dataset("eli5", split="train_asks[:5000]")

eli5=eli5.train_test_split(test_size=0.2)

def preprocess_function(example):
  return tokenizer([" ".join(x) for x in example["answers.text"]])


def group_texts(examples):
    concatenated_examples={k: sum(examples[k], []) for k in examples.keys()}
    total_length=len(concatenated_examples[list(examples.keys())[0]])
    if total_length>=block_size:
        total_length=(total_length//block_size)* block_size
    # Split by chunks of block size
    result={
        k: [t[i: i+block_size] for i in range(0, total_length, block_size)]
        for k,t in concatenated_examples.items()
    }

    result["labels"]=result["input_ids"].copy()
    return result

tokenized_eli5=eli5.map(preprocess_function,
                        batched=True,
                        num_proc=4,
                        remove_columns=eli5["train"].column_names)

lm_dataset=tokenized_eli5.map(group_texts, batched=True, num_proc=4)