import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import model,tokenizer


# Streamlit app
def main():
    st.title("CausalLM Text Generation App")

    # Text input for user prompt
    prompt = st.text_input("Enter your prompt here:")

    # Generate text when button is clicked
    if st.button("Generate Text"):
        if prompt:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            # Generate text
            output = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

            # Decode and display generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.text_area("Generated Text:", value=generated_text, height=200)

if __name__ == "__main__":
    main()
