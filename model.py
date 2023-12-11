import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class DNASequenceTransformer(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DNASequenceTransformer, self).__init__()
        # Use a randomly initialized transformer model
        config = BertConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2)
        self.encoder = BertModel(config=config)

        # Custom embedding layer for DNA physical properties
        self.embedding = nn.Linear(
            num_features, self.encoder.config.hidden_size * num_features
        )

        # Classification head
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

        # Apply sigmoid for BCELoss
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is your input tensor with shape [batch_size, seq_len, num_features]

        # Convert physical properties to embeddings
        embeddings = self.embedding(x)

        # Reshape embeddings to [batch_size, seq_len, hidden_size]
        embeddings = embeddings.view(
            embeddings.shape[0], -1, self.encoder.config.hidden_size
        )

        # Process embeddings through the transformer
        encoded_layers = self.encoder(inputs_embeds=embeddings)

        # Use the output of the last layer for classification
        sequence_output = encoded_layers.last_hidden_state
        pooled_output = sequence_output[:, 0]

        # Classification
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)
        return logits


class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)  # Apply sigmoid for BCELoss
        return out
