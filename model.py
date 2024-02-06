from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline

tokenizer=AutoTokenizer.from_pretrained("big_tits/gpt2")

model=AutoModelForCausalLM.from_pretrained("big_tits/gpt2")

if __name__=="__main__":

    prompt="Somatic hypermutation allows the immune system to"

    generator=pipeline("text-generation" , model=model , tokenizer=tokenizer)

    print(generator(prompt))
