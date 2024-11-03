import open_clip
import torch

def download_bioclip_model(save_name: str):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    torch.save(model.state_dict(), f"{save_name}_model_weights.pt")
    torch.save(preprocess_train, f"{save_name}_preprocess_train.pt")
    torch.save(preprocess_val, f"{save_name}_preprocess_val.pt")
    torch.save(tokenizer, f"{save_name}_tokenizer.pt")
    



if __name__ == "__main__":
    download_bioclip_model("bio_clip")