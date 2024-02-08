from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from model import model,data_collator
from dataset import lm_dataset

training_args=TrainingArguments(
    output_dir="result",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="wandb",
    push_to_hub=False,
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

if __name__=="__main__":
    trainer.train()