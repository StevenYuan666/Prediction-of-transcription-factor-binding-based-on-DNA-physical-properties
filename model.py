import torch
import torch.nn as nn
from transformers import BertModel

class DNASequenceTransformer(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DNASequenceTransformer, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        # Custom embedding layer for DNA physical properties
        self.embedding = nn.Linear(num_features, self.encoder.config.hidden_size)

        # Classification head
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x):
        # x is your input tensor with shape [batch_size, seq_len, num_features]

        # Convert physical properties to embeddings
        embeddings = self.embedding(x)

        # Process embeddings through the transformer
        encoded_layers = self.encoder(inputs_embeds=embeddings)

        # Use the output of the last layer for classification
        sequence_output = encoded_layers.last_hidden_state
        pooled_output = sequence_output[:, 0]

        # Classification
        logits = self.classifier(pooled_output)
        return logits
