import torch
from torch.utils.data import DataLoader, Dataset


class DNAPropertiesDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # Convert labels to integers (0 for 'negative', 1 for 'bound')
        self.dataframe["label"] = self.dataframe["label"].map(
            {"negative": 0, "bound": 1}
        )

        # Separate features and labels
        self.features = self.dataframe.drop("label", axis=1).values
        self.labels = self.dataframe["label"].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Convert the features and labels to torch tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


def get_data_loader(df, batch_size=32, shuffle=True):
    # Create an instance of the custom dataset
    dataset = DNAPropertiesDataset(df)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


if __name__ == "__main__":
    from classifier_training import preprocess

    tf = "TBP"
    df_properties_both = preprocess(tf)
    data_loader = get_data_loader(df_properties_both)
    print(data_loader)
    print(len(data_loader))
    for features, labels in data_loader:
        print(features.shape)
        print(labels.shape)
        break
