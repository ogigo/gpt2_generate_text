from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM

model_name="distilgpt2"
tokenizer=AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token=tokenizer.eos_token
# Use the end of sequence token as the padding token and set `mlm=False`.
# This will use the inputs as labels shifted to the right by one element.
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model=AutoModelForCausalLM.from_pretrained("distilgpt2")