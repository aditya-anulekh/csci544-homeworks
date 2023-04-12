import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100,
                 word_embeddings=None, **kwargs):
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size

        if word_embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(word_embeddings,
                                                          freeze=True,
                                                          padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # +1 for case_bool
            hidden_size=kwargs.get('hidden_size', 256),
            num_layers=kwargs.get('num_layers', 1),
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=kwargs.get('dropout', 0.33))
        self.linear = nn.Linear(
            in_features=2*kwargs.get('hidden_size', 256),
            out_features=128
        )
        # self.classifier = nn.Linear(
        #     in_features=128,
        #     out_features=self.tagset_size
        # )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=self.tagset_size),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=self.tagset_size)
        )
        self.elu = nn.ELU()

    def forward(self, sentences, case_bool, lengths):
        # Get embeddings for sentences and create a PackedSequence
        x = self.embedding(sentences)  # (batch_size, seq_length, 100)

        # (batch_size, seq_length) -> (batch_size, seq_length, 1)
        case_bool = torch.unsqueeze(case_bool, dim=2)

        x = torch.cat([x, case_bool], dim=2)  # (batch_size, seq_length, emb+1)
        x = pack_padded_sequence(x, lengths, batch_first=True,
                                 enforce_sorted=False)

        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)

        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)

        return x
