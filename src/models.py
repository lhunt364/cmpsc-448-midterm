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
        filter_scores = torch.relu(self.conv(embedded)) # (batch, num_filters, number of windows = (seq_len + 2) - kernel_size + 1) -> slide filters along sequence, one score per position
        pooled = filter_scores.max(dim=2).values # (batch, num_filters) each filter's max match in the text
        pooled = self.dropout(pooled)
        class_logits = self.classifier(pooled) # (batch, num_classes) each class's score for each sequence
        return class_logits


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.5, num_classes=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_to_gates = torch.nn.Linear(embed_dim, hidden_dim * 4)
        self.hidden_to_gates = torch.nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids):
        # token_ids -> (batch, sequence_length)
        batch_size, seq_len = token_ids.shape
        embedded = self.embedding(token_ids) # (batch, seq_len, embed_dim)
        is_real_token = (token_ids != 0).float().unsqueeze(2) # (batch, seq_len, 1) if not <pad>, otherwise (batch, seq_len, 0)
        hidden = torch.zeros(batch_size, self.hidden_dim, device=token_ids.device) # (batch, hidden_dim)
        cell = torch.zeros(batch_size, self.hidden_dim, device=token_ids.device) # (batch, hidden_dim)

        input_part_all_steps = self.input_to_gates(embedded) # (batch, seq_len, hidden_dim * 4)

        for step in range(seq_len):
            gate_values = input_part_all_steps[:, step, :] + self.hidden_to_gates(hidden) # (batch, hidden_dim * 4)
            input_gate, forget_gate, output_gate, candidate = gate_values.chunk(4, 1)
            input_gate = torch.sigmoid(input_gate) # each are (batch, hidden_dim)
            forget_gate = torch.sigmoid(forget_gate)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.tanh(candidate)

            new_cell = forget_gate * cell + input_gate * candidate
            new_hidden = output_gate * torch.tanh(new_cell)

            mask = is_real_token[:, step, :] # use new_val when token is real, keep current val if not
            cell = mask * new_cell + (1 - mask) * cell
            hidden = mask * new_hidden + (1 - mask) * hidden

        class_logits = self.classifier(self.dropout(hidden))
        return class_logits


if __name__ == "__main__":
    vocab_size = 25000
    for model in [CNN(vocab_size), LSTM(vocab_size)]:
        class_scores = model(torch.randint(0, vocab_size, (64, 256)))
        print(type(model).__name__)
        print("output shape:", tuple(class_scores.shape))
        print("number of learnable parameters:", sum(p.numel() for p in model.parameters()))
        print()
