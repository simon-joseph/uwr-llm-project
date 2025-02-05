import torch
import torch.nn as nn
from PIL import Image


# Define the mapping network (prefix transformer)
class MappingNetwork(nn.Module):
    def __init__(self, clip_embedding_dim, prefix_length, gpt2_embedding_dim):
        """
        Args:
            clip_embedding_dim: Dimension of the CLIP image embedding.
            prefix_length: Number of tokens to use as prefix for GPT-2.
            gpt2_embedding_dim: GPT-2 token embedding dimension.
        """
        super(MappingNetwork, self).__init__()
        self.prefix_length = prefix_length
        self.gpt2_embedding_dim = gpt2_embedding_dim
        # A simple two-layer MLP
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
        # clip_embedding: (batch_size, clip_embedding_dim)
        mapped = self.fc(
            clip_embedding
        )  # (batch_size, prefix_length * gpt2_embedding_dim)
        # Reshape to (batch_size, prefix_length, gpt2_embedding_dim)
        mapped = mapped.view(-1, self.prefix_length, self.gpt2_embedding_dim)
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


def generate_caption(
    PIL_image,
    clip_model,
    clip_processor,
    gpt2_model,
    gpt2_tokenizer,
    mapping_network,
    prefix_length,
    device,
):
    # Load and preprocess the image
    image = PIL_image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract CLIP image features
    with torch.no_grad():
        clip_outputs = clip_model.get_image_features(
            **inputs
        )  # shape: (batch_size, clip_embedding_dim)
        # Optionally, normalize the embedding:
        clip_embedding = clip_outputs / clip_outputs.norm(dim=-1, keepdim=True)

    # Map the CLIP embedding to a GPT-2 prefix embedding
    prefix_embedding = mapping_network(
        clip_embedding
    )  # shape: (batch_size, prefix_length, gpt2_embedding_dim)

    # print decoded embeddings
    print("How mapped embeddings look like:")
    print(
        gpt2_tokenizer.decode(
            prefix_embedding[0].argmax(dim=1).tolist(), skip_special_tokens=True
        )
    )
    print("-" * 50)

    # Prepare a prompt (can be empty) for GPT-2
    prompt = " : "
    tokens = gpt2_tokenizer(prompt, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    # Get GPT-2 embeddings for the prompt tokens
    print(tokens["input_ids"])
    gpt2_inputs_embeds = gpt2_model.transformer.wte(
        tokens["input_ids"]
    )  # shape: (batch_size, seq_len, gpt2_embedding_dim)

    # Concatenate the prefix embedding and GPT-2 prompt embeddings
    inputs_embeds = torch.cat((prefix_embedding, gpt2_inputs_embeds), dim=1)

    inputs_embeds = prefix_embedding

    # Generate a caption using GPT-2
    generated_outputs = gpt2_model.generate(
        inputs_embeds=inputs_embeds,
        max_length=prefix_length + 50,  # allow generation of 50 tokens after the prefix
        do_sample=True,
        num_beams=4,
        temperature=0.95,
        top_p=0.9,
        eos_token_id=gpt2_tokenizer.eos_token_id,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )

    # Remove the prefix tokens from the generated output and decode
    generated_tokens = generated_outputs[0][prefix_length:]
    caption = gpt2_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return caption


# if __name__ == "__main__":
# # test the MappingNetwork
# clip_embedding_dim = 512
# prefix_length = 10
# gpt2_embedding_dim = 768
# mapping_network = MappingNetwork(
#     clip_embedding_dim, prefix_length, gpt2_embedding_dim
# )
# print(mapping_network)
# clip_embedding = torch.randn(2, clip_embedding_dim)
# mapped = mapping_network(clip_embedding)
# print(mapped.shape)  # torch.Size([2, 10, 768])
