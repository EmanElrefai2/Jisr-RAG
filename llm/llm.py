from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

from helpers.config import config
from helpers.utils import extract_references
from helpers.logger import create_logger

logger = create_logger(__name__, os.getenv("LOGGING_LEVEL", "INFO"))


class LLM:
    def initialize_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        logger.info("LLM model has been initialized")

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        logger.info("LLM tokenizer has been initialized")

        return model, tokenizer

    def generate_response(model, tokenizer, mood: str, query: str, 
                          documents: list):
        try:
            refrences = []

            if mood == "rag":
                sys_prompt = config.RAG_SYSTEM_PROMPT.format(query=query)
                print(sys_prompt)
                text = ""
                for i in range(len(documents)):
                    text += f"Document {i}:\n"
                    text += f"{documents[i]['page_content']}\n\n"
            else:
                sys_prompt = config.SUMM_SYSTEM_PROMPT.format(title=query)
                text = ""
                for i in range(len(documents)):
                    text += f"Document {i}:\n"
                    text += f"{documents[i]['meta_data']['Section Title']}\n"
                    text += f"{documents[i]['page_content']}\n\n"

            model_inputs = tokenizer([
                config.ALPACA_PROMPT.format(
                    sys_prompt,
                    text,
                    "",
                )
            ], return_tensors="pt").to(config.DEVICE)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024, no_repeat_ngram_size=2,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            if tokenizer.decode([generated_ids[0][-1].item()]) != "<|endoftext|>":
                logger.info("Hallucination detected")
                response = "لا يمكنني العثور على الإجابة."
                return response, refrences, ""

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            extracted_refrences, response = extract_references(response)
            print(response)
            print("##########################################")

            if len(refrences) == 0:
                refrences = [
                    doc["meta_data"]["chunk_id"]
                    for doc in documents
                ]

            del model_inputs
            del generated_ids
            torch.cuda.empty_cache()

            return response, refrences, documents[0]["meta_data"]["Document_id"]

        except Exception as e:
            logger.error(str(e))
            response = "لا يمكنني العثور على الإجابة."
            return response, [], ""
