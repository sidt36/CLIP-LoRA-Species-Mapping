import open_clip
import torch

def download_bioclip_model(save_name: str):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    torch.save(model.state_dict(), "model_weights.pt")
    torch.save(tokenizer, "preprocess_train.pt")



if __name__ == "__main__":
    
    download_bioclip_model("bio_clip")