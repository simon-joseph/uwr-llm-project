import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as func
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

PREFIX_LENGTH = 60

# Load RSICD dataset
dataset = load_dataset("arampacha/rsicd", split="train")

# Load CLIP and GPT-2 models
device = torch.device("mps")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


# Define the mapping network
class MappingNetwork(nn.Module):
    def __init__(self, clip_embedding_dim, prefix_length, gpt2_embedding_dim):
        super(MappingNetwork, self).__init__()
        self.prefix_length = prefix_length
        self.gpt2_embedding_dim = gpt2_embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(
                clip_embedding_dim,
                (clip_embedding_dim + prefix_length * gpt2_embedding_dim) // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                (clip_embedding_dim + prefix_length * gpt2_embedding_dim) // 2,
                prefix_length * gpt2_embedding_dim,
            ),
        )

    def forward(self, clip_embedding):
        mapped = self.fc(
            clip_embedding
        )  # (batch_size, prefix_length * gpt2_embedding_dim)
        mapped = mapped.view(
            -1, self.prefix_length, self.gpt2_embedding_dim
        )  # Reshape for GPT-2
        return mapped


class MappingNetwork2(nn.Module):
    def __init__(self, clip_embedding_dim, prefix_length, gpt2_embedding_dim):
        super(MappingNetwork2, self).__init__()
        self.prefix_length = prefix_length
        self.gpt2_embedding_dim = gpt2_embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(
                clip_embedding_dim,
                (clip_embedding_dim + prefix_length * gpt2_embedding_dim) // 2,
            ),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(
                (clip_embedding_dim + prefix_length * gpt2_embedding_dim) // 2,
                prefix_length * gpt2_embedding_dim,
            ),
            nn.Dropout(0.1),
        )

    def forward(self, clip_embedding):
        mapped = self.fc(
            clip_embedding
        )  # (batch_size, prefix_length * gpt2_embedding_dim)
        mapped = mapped.view(
            -1, self.prefix_length, self.gpt2_embedding_dim
        )  # Reshape for GPT-2
        return mapped


# Custom dataset class for training
class SatelliteImageCaptionDataset(Dataset):
    def __init__(self, dataset, clip_processor, tokenizer, prefix_length):
        self.data = dataset
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]  # Direct PIL image
        caption = item["captions"][0]  # First caption

        # Convert image to CLIP features
        clip_inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
        clip_embedding = clip_model.get_image_features(**clip_inputs).squeeze(
            0
        )  # .to(device)  # Shape: (512,)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50,
        )
        input_ids = tokens["input_ids"].squeeze(0)  # Shape: (max_length,)

        return clip_embedding, input_ids


# Initialize dataset and dataloader
train_dataset = SatelliteImageCaptionDataset(
    dataset, clip_processor, gpt2_tokenizer, prefix_length=PREFIX_LENGTH
)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize mapping network and optimizer
mapping_network = MappingNetwork2(
    clip_embedding_dim=512, prefix_length=PREFIX_LENGTH, gpt2_embedding_dim=768
).to(device)
optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    progress = tqdm(total=len(train_dataloader))
    for clip_embeddings, captions in train_dataloader:
        clip_embeddings, captions = clip_embeddings.to(device), captions.to(device)
        # Generate GPT-2 prefixes (batch_size, prefix_length, gpt2_embedding_dim)
        prefixes = mapping_network(clip_embeddings)

        # Get actual sequence length of each caption
        caption_lengths = (captions != gpt2_tokenizer.pad_token_id).sum(dim=1)

        # Ensure all captions are at least prefix_length long
        max_caption_length = captions.shape[1]
        total_length = min(
            max_caption_length + prefixes.shape[1], 50
        )  # GPT-2 max input size

        # Prepare input tensors with correct size
        gpt2_inputs = torch.cat(
            [
                torch.zeros(
                    captions.shape[0],
                    prefixes.shape[1],
                    dtype=torch.long,
                    device=device,
                ),
                captions,
            ],
            dim=1,
        )  # (batch_size, prefix_length + caption_length)
        gpt2_inputs = gpt2_inputs[
            :, :total_length
        ]  # Ensure consistency in sequence length

        # Convert to GPT-2 embeddings
        gpt2_inputs_embeds = gpt2_model.transformer.wte(
            gpt2_inputs
        )  # (batch_size, total_length, gpt2_embedding_dim)

        # Forward pass through GPT-2
        outputs = gpt2_model(inputs_embeds=gpt2_inputs_embeds, labels=gpt2_inputs)

        # Compute loss
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.set_postfix({"loss": loss.item()})
        progress.update()

    progress.close()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

# Save the trained mapping network
torch.save(mapping_network.state_dict(), "mapping_network2-2.pth")
print("Training complete!")
