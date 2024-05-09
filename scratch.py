import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# TODO: Add label mixing to the generator and class discriminator to interpolate
#  the decision boundaries between classes.
# TODO: Add GPU support.
# TODO: Incorporate timestep information into the model.

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, num_labels):
        super(ConditionalGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_labels = num_labels

        # Embeddings for mean and deviation
        self.mean_embedding = nn.Embedding(num_labels, z_dim)
        self.dev_embedding = nn.Embedding(num_labels, z_dim)

        # Initialise embeddings
        self.mean_embedding.weight.data.normal_(0, 0.02)
        self.dev_embedding.weight.data.normal_(1, 0.02)

        self.dropout1 = nn.Dropout(p=0.1)
        # Generator network layers
        self.W1 = nn.Parameter(torch.randn(z_dim, hidden_dim * 2))
        self.W1 = nn.init.xavier_uniform_(self.W1)
        self.b1 = nn.Parameter(torch.randn(hidden_dim * 2))
        self.dropout2 = nn.Dropout(p=0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim * 2, hidden_dim))
        self.W2 = nn.init.xavier_uniform_(self.W2)
        self.b2 = nn.Parameter(torch.randn(hidden_dim))
        self.dropout3 = nn.Dropout(p=0.1)
        self.W3 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.W3 = nn.init.xavier_uniform_(self.W3)
        self.b3 = nn.Parameter(torch.randn(output_dim))

        self.softplus = nn.Softplus()

    def forward(self, z, labels):
        # Get mean and deviation embeddings for the given labels
        mean = self.mean_embedding(labels)
        dev = self.dev_embedding(labels)

        # Modulate the input noise
        z = mean + z * dev
        z = self.dropout1(z)
        # Pass the modulated noise through the network
        x = torch.matmul(z, self.W1) + self.b1
        x = F.relu(x)
        # x = self.bn1(x)
        x = self.dropout2(x)
        x = torch.matmul(x, self.W2) + self.b2
        x = F.relu(x)
        # x = self.bn2(x)
        x = self.dropout3(x)
        x = torch.matmul(x, self.W3) + self.b3
        x = self.softplus(x)
        return x


class ClassDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ClassDiscriminator, self).__init__()
        self.dropout1 = nn.Dropout(p=0.1)
        # Initialise weights
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim*2, kernel_size=input_dim,
            stride=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=1,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(p=0.1)
        self.W1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W1 = nn.init.xavier_uniform_(self.W1)
        self.b1 = nn.Parameter(torch.randn(hidden_dim))
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(p=0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, num_classes))
        self.W2 = nn.init.xavier_uniform_(self.W2)
        self.b2 = nn.Parameter(torch.randn(num_classes))
        self.bn4 = nn.BatchNorm1d(num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(-2)
        x = self.dropout1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout2(x)
        # reshape conv output so that channels are the last dimension
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout3(x)
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, self.W1) + self.b1
        x = F.relu(x)
        x = x.squeeze(-2)
        x = self.bn3(x)
        x = self.dropout4(x)
        x = torch.matmul(x, self.W2) + self.b2
        x = self.bn4(x)
        x = self.softmax(x)
        return x


class AdversarialDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdversarialDiscriminator, self).__init__()
        self.dropout1 = nn.Dropout(p=0.1)
        # Initialise weights
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim*2, kernel_size=input_dim,
            stride=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=1,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(p=0.1)
        self.W1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W1 = nn.init.xavier_uniform_(self.W1)
        self.b1 = nn.Parameter(torch.randn(hidden_dim))
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(p=0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, 1))
        self.W2 = nn.init.xavier_uniform_(self.W2)
        self.b2 = nn.Parameter(torch.randn(1))
        self.bn4 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = x.unsqueeze(-2)
        x = self.dropout1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout3(x)
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, self.W1) + self.b1
        x = F.relu(x)
        x = x.squeeze(-2)
        x = self.bn3(x)
        x = self.dropout4(x)
        x = torch.matmul(x, self.W2)
        x = self.bn4(x)
        return x


def measure_sparsity(output, threshold=1e-5):
    """
    Measures the sparsity of a tensor by calculating the percentage of elements
    that are near zero (below a certain threshold).

    :params output:
        The output tensor from the neural network.
    :params threshold:
        A threshold to determine if an element is considered zero.

    :return:
        The proportion of elements in the output tensor that are below the
        threshold.
    """
    # Count elements that are approximately zero
    near_zero = (output.abs() < threshold).float()
    sparsity = near_zero.mean()

    return sparsity.item()

class CiteDataset(Dataset):
    def __init__(self, data, metadata, label_to_idx):
        self.data = data
        self.metadata = metadata
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cell_id = self.metadata.iloc[idx]['cell_id']
        cell_type = self.metadata.iloc[idx]['cell_type']
        cell_type = self.label_to_idx[cell_type]
        features = self.data.loc[cell_id]
        return features, cell_type


def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return torch.tensor(data), torch.tensor(labels)


df_cite = pd.read_hdf(
    '../data/train_cite_inputs.h5', key='train_cite_inputs'
)

metadata = pd.read_csv('../data/metadata.csv')
# exclude all celltype == hidden
metadata = metadata[metadata['cell_type'] != 'hidden']

# any(x in df_cite.index for x in metadata['cell_id'])

# cell_ids = [x for x in metadata['cell_id'] if x in df_cite.index]
metadata_cell_ids = list(metadata['cell_id'])
cell_ids = [x for x in df_cite.index if x in metadata_cell_ids]

filtered_metadata = metadata[metadata['cell_id'].isin(cell_ids)].reset_index(
    drop=True
)
filtered_df_cite = df_cite.loc[cell_ids]

# Create the generator and discriminator
z_dim = 256
hidden_dim = 256
output_dim = len(df_cite.columns)

label_to_idx = {
    label: idx for idx, label in enumerate(metadata['cell_type'].unique())
}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

num_labels = len(label_to_idx.keys())

n_epochs = 5
display_step = 1
batch_size = 128

# Sparsity parameters
sparsity_target = 0.9
sparsity_weight = 0.1
l1_lambda = 0.00001

dataset = CiteDataset(filtered_df_cite, filtered_metadata, label_to_idx)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

generator = ConditionalGenerator(z_dim, hidden_dim, output_dim, num_labels)
class_discriminator = ClassDiscriminator(output_dim, hidden_dim, num_labels)
adv_discriminator = AdversarialDiscriminator(output_dim, hidden_dim)

# Create optimisers for the generator and discriminators
gen_opt = optim.Adam(generator.parameters(), lr=0.0001)
class_disc_opt = optim.Adam(class_discriminator.parameters(), lr=0.0001)
adv_disc_opt = optim.Adam(adv_discriminator.parameters(), lr=0.0001)

# Loss functions
multi_class_criterion = nn.CrossEntropyLoss()
binary_class_criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(n_epochs):
    for i, (real, labels) in enumerate(dataloader):
        real = real.float()  # Ensure data is in float format if not already
        batch_size = real.size(0)

        # Generate fake data
        fake_noise = torch.randn(batch_size, z_dim)
        fake = generator(fake_noise, labels)
        sparsity = measure_sparsity(fake)
        sparsity_penalty = abs(sparsity - sparsity_target) * sparsity_weight

        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Update class discriminator (Multi-class classification)
        class_disc_opt.zero_grad()
        disc_real_pred = class_discriminator(real)
        class_disc_loss = multi_class_criterion(disc_real_pred, labels)
        class_disc_loss.backward()
        class_disc_opt.step()

        # Update adversarial discriminator (Real/Fake classification)
        adv_disc_opt.zero_grad()
        real_pred = adv_discriminator(real)
        # Detach to stop gradients to generator
        fake_pred = adv_discriminator(fake.detach())
        adv_real_loss = binary_class_criterion(real_pred, real_labels)
        adv_fake_loss = binary_class_criterion(fake_pred, fake_labels)
        adv_loss = (adv_real_loss + adv_fake_loss) / 2
        adv_loss.backward()
        adv_disc_opt.step()

        # Update generator (both for class and to fool adversarial
        # discriminator)
        gen_opt.zero_grad()
        fake_pred = adv_discriminator(fake)
        # Try to fool the adversarial discriminator
        adv_gen_loss = binary_class_criterion(fake_pred, real_labels)
        disc_fake_pred = class_discriminator(fake)
        gen_class_loss = multi_class_criterion(disc_fake_pred, labels)
        # l1 loss to encourage sparsity
        l1_norm = sum(p.abs().sum() for p in generator.parameters()) * l1_lambda
        gen_loss = adv_gen_loss + gen_class_loss + l1_norm + sparsity_penalty
        # print("Adv Loss:", adv_loss.item())
        # print("Gen Class Loss:", gen_class_loss.item())
        # print("L1 Penalty:", l1_norm.item())
        # print("Sparsity Penalty:", sparsity_penalty)
        gen_loss.backward()
        gen_opt.step()

        # Logging for tracking progress (optional)
        if (i + 1) % display_step == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Step [{i+1}/{len(dataloader)}], "
                  f"Real/Fake_Disc_Loss: {adv_loss.item():.4f}, "
                  f"Gen_Loss: {gen_loss.item():.4f}, "
                  f"Class_classifier_Loss: {class_disc_loss.item():.4f}")

