import sys
sys.path.append(".")
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # you can also use other GPU devices or multiple GPU devices
import torch
from llama_models.models.llama3.reference_impl.generation import Llama
from llama_models.models.llama3.api.chat_format import ModelInput
from torch.nn.utils.rnn import pad_sequence
from glob import glob
import argparse

''' Given a folder, this script extracts the sentence embeddings for all the txt files in the folder (and its subfolder) and saves them as .pt files.
    it pads the sentence embeddings for txt files under each subfolder
    since the default Llama model runs in a distributed fashion, you should run this script with "torchrun scripts/extract_language_embedding.py"
'''

def process_txt_files_in_subdirectories(folder_path):
    # iterate through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        txt_files = [f for f in files if f.endswith('.txt')]
        
        # if there are txt files in the current subdirectory. This is the most bottom level of the directory tree
        if txt_files:
            all_sentence_embeddings = []
            for txt_file in txt_files:
                txt_file_path = os.path.join(root, txt_file)

                with open(txt_file_path, "r") as f:
                    sentence = f.read()
                encoded_input = llama.tokenizer.encode(sentence, bos=True, eos=False)
                with torch.inference_mode():
                    tokens = torch.tensor([encoded_input], dtype=torch.long, device="cuda")
                    logits = llama.model.forward(tokens, 0)

                sentence_embedding = h_embedding.clone()    # use the last hidden layer's output as the sentence embedding
                print(sentence_embedding.shape)
                all_sentence_embeddings.append(sentence_embedding[0])   # [len, 4096]
                
            # padding
            padded_sentence_embeddings = pad_sequence(all_sentence_embeddings, batch_first=True, padding_value=0)

            # save a embedding file for each txt file
            for i in range(len(txt_files)):
                txt_file_path = os.path.join(root, txt_files[i])
                file_name = txt_file_path.split("/")[-1].split(".")[0]
                torch.save(padded_sentence_embeddings[i].cpu(), os.path.join(os.path.dirname(txt_file_path), file_name+"_sentence_embedding.pt"))
                print(f"Saved sentence embedding for {txt_file_path} at {os.path.join(os.path.dirname(txt_file_path), file_name+'_sentence_embedding.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/tracked_results/libero_10")
    args = parser.parse_args()
        
    ckpt_dir = "Meta-Llama-3.1-8B"
    tokenizer_path = "Meta-Llama-3.1-8B/tokenizer.model"
    max_seq_len = 512
    max_batch_size = 1
    model_parallel_size = None

    def hook_fn(module, input, output):
        global h_embedding
        h_embedding = output

    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    llama.model = llama.model.to("cuda")

    handle = llama.model.layers[-1].register_forward_hook(hook_fn)

    process_txt_files_in_subdirectories(args.data_dir)

