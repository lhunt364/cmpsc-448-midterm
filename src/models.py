import torch

# step 2.5 also, define model behavior

class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, kernel_size=5, dropout=0.5, num_classes=4):
        super().__init__()
        self.embedded = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0) # a (vocab_size, embed_dim) tensor mapping each vocab word to a 128 value vector
        self.conv = torch.nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(num_filters, num_classes)

    def forward(self, token_ids):
        # token_ids -> (batch, sequence_length)
        embedded = self.embedded(token_ids) # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2) # (batch, embed_dim, seq_len) because that's how a Conv1d wants it
        filter_scores = torch.relu(self.conv(embedded)) # (batch, num_filters, number of windows = seq_len - kernel_size + 1) -> slide filters along sequence, one score per position
        pooled = filter_scores.max(dim=2).values # (batch, num_filters) each filter's max match in the text
        class_logits = self.classifier(pooled) # (batch, num_classes) each class's score for each sequence
        return class_logits


if __name__ == "__main__":
    vocab_size = 30000
    model = CNN(vocab_size)
    class_scores = model(torch.randint(0, vocab_size, (64, 256)))
    print("output shape:", tuple(class_scores.shape))
    print("number of learnable parameters:", sum(p.numel() for p in model.parameters()))
