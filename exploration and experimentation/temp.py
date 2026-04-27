
# %%
import pandas as pd
import numpy as np

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

df = pd.read_csv("nsl-kdd/KDDTrain+.txt", header=None, names=columns)
print(df.shape)
print(df.head())

# %%
print(df.dtypes)

# %%
selected_features = [
    "duration",           # how long the connection lasted
    "src_bytes",          # bytes sent from source
    "dst_bytes",          # bytes sent to destination
    "wrong_fragment",     # unusual fragmentation → suspicious
    "urgent",             # urgent packets → suspicious
    "hot",                # "hot" indicators inside connection
    "num_failed_logins",  # failed login attempts
    "num_compromised",    # compromised conditions
    "count",              # connections to same host in last 2 sec
    "srv_count",          # connections to same service in last 2 sec
    "serror_rate",        # % SYN errors → DoS indicator
    "rerror_rate",        # % REJ errors
    "same_srv_rate",      # % connections to same service
    "diff_srv_rate",      # % connections to different services → scanning
    "dst_host_count"      # connections to same destination host
]

X = df[selected_features].copy()
print(X.shape)  # should be (125973, 15)

# %%
# Convert label to binary: normal=0, attack=1
y = (df["label"] != "normal").astype(int).values
print(f"Normal: {(y==0).sum()}, Attack: {(y==1).sum()}")

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled.shape)   # (125973, 15)
print(X_scaled.mean(axis=0).round(2))  # should be ~0 for all
print(X_scaled.std(axis=0).round(2))   # should be ~1 for all

# %%
np.save("X_nodes.npy", X_scaled)
np.save("y_labels.npy", y)

print("Saved X_nodes.npy and y_labels.npy")

# %%
# Make sure you're using the full dataset
df = pd.read_csv("nsl-kdd/KDDTrain+_20Percent.txt", header=None, names=columns)
print(df.shape)  # should be ~25000, not 2000

# %%
import torch
from collections import defaultdict

edges = []
MAX_EDGES_PER_NODE = 5

# Group by protocol_type + service combined — much more specific
service_groups = defaultdict(list)
for idx, row in df.iterrows():
    key = (row["protocol_type"], row["service"])  # more specific grouping
    service_groups[key].append(idx)

print(f"Number of groups: {len(service_groups)}")
# Should be 100+ groups now, much more meaningful structure

for group, indices in service_groups.items():
    for i, node in enumerate(indices):
        neighbors = indices[i+1 : i+1+MAX_EDGES_PER_NODE]
        for neighbor in neighbors:
            edges.append([node, neighbor])
            edges.append([neighbor, node])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"Total edges: {edge_index.shape[1]}")
print(f"Avg per node: {edge_index.shape[1] / len(df):.2f}")

# %%
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(edge_index.shape)  # [2, num_edges]
print(edge_index)
# tensor([[  0,   1,   1,   0, ...],
#         [  1,   0,   3,   5, ...]])

# %%
from torch_geometric.data import Data

# Get features and labels for the subset
X_small = X_scaled[df_small.index]
y_small = y[df_small.index]

x_tensor = torch.tensor(X_small, dtype=torch.float)
y_tensor = torch.tensor(y_small, dtype=torch.long)

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
print(data)
# Data(x=[2000, 15], edge_index=[2, ~N], y=[2000])

# %%
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Filter edges where BOTH nodes are < 50
mask = (edge_index[0] < 50) & (edge_index[1] < 50)
small_edge_index = edge_index[:, mask]

small_data = Data(
    x=x_tensor[:50],
    edge_index=small_edge_index,
    y=y_tensor[:50]
)

G = to_networkx(small_data, to_undirected=True)

# ✅ Use actual number of nodes in G, not hardcoded 50
num_nodes = G.number_of_nodes()
print(f"Actual nodes in G: {num_nodes}")

colors = ["red" if y_tensor[i] == 1 else "skyblue" for i in range(num_nodes)]

plt.figure(figsize=(10, 7))
nx.draw(G, node_color=colors, node_size=200,
        with_labels=False, edge_color="gray", alpha=0.8)
plt.title("NSL-KDD Subgraph — Skyblue=Normal, Red=Attack")
plt.show()

# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# %%
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(input_dim=15, hidden_dim=32, output_dim=2)
print(model)

# %%
num_nodes = data.num_nodes
indices = torch.randperm(num_nodes)  # shuffle

train_size = int(0.7 * num_nodes)
val_size   = int(0.15 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[indices[:train_size]]                      = True
val_mask[indices[train_size:train_size+val_size]]     = True
test_mask[indices[train_size+val_size:]]              = True

data.train_mask = train_mask
data.val_mask   = val_mask
data.test_mask  = test_mask

print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = correct / mask.sum()
    return acc.item()

# Run training
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc = evaluate(data.train_mask)
        val_acc   = evaluate(data.val_mask)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# %%
test_acc = evaluate(data.test_mask)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# %%
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)

y_true = data.y[data.test_mask].numpy()
y_pred = pred[data.test_mask].numpy()

print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
print(confusion_matrix(y_true, y_pred))

# %%
# Run these and share the outputs
print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
print(confusion_matrix(y_true, y_pred))

# Also share these
print(f"Class distribution in full dataset:")
print(f"Normal: {(data.y == 0).sum().item()}")
print(f"Attack: {(data.y == 1).sum().item()}")

print(f"\nEdge info:")
print(f"Num nodes: {data.num_nodes}")
print(f"Num edges: {data.num_edges}")
print(f"Avg edges per node: {data.num_edges / data.num_nodes:.2f}")

# %%



