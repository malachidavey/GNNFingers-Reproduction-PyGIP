import random
import warnings

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

from pygip.models.nn.backbones import GCN
from .base import BaseAttack

warnings.filterwarnings('ignore')


class DGLEdgePredictor(nn.Module):
    """DGL version of edge prediction module"""

    def __init__(self, input_dim, hidden_dim, num_classes, device):
        super(DGLEdgePredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device

        # Use the same GCN backbone as the target model
        self.gnn = GCN(input_dim, hidden_dim)
        self.node_classifier = nn.Linear(hidden_dim, num_classes)

        # Edge prediction layer
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, graph, features):
        # Get node embeddings
        node_embeddings = self.gnn(graph, features)

        # Node classification
        node_logits = self.node_classifier(node_embeddings)

        return node_embeddings, node_logits

    def predict_edges(self, node_embeddings, node_pairs):
        """Predict edge existence probability"""
        if len(node_pairs) == 0:
            return torch.tensor([], device=self.device)

        node_pairs = torch.tensor(node_pairs, device=self.device)
        src_embeddings = node_embeddings[node_pairs[:, 0]]
        dst_embeddings = node_embeddings[node_pairs[:, 1]]

        # Concatenate source and destination node embeddings
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        edge_probs = self.edge_predictor(edge_features).squeeze()

        return edge_probs


class DGLSurrogateModel(nn.Module):
    """DGL version of surrogate model"""

    def __init__(self, input_dim, num_classes, model_type='GCN'):
        super(DGLSurrogateModel, self).__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.gnn = GCN(input_dim, num_classes)
        else:
            # Can extend to other model types
            self.gnn = GCN(input_dim, num_classes)

    def forward(self, graph, features):
        return self.gnn(graph, features)


class RealisticAttack(BaseAttack):
    """DGL-based GNN model extraction attack"""
    supported_api_types = {"dgl"}
    supported_datasets = {}

    def __init__(self, dataset, attack_node_fraction: float, model_path: str = None,
                 hidden_dim: int = 64, threshold_s: float = 0.7, threshold_a: float = 0.5):
        super().__init__(dataset, attack_node_fraction, model_path)

        self.hidden_dim = hidden_dim
        self.threshold_s = threshold_s  # Cosine similarity threshold
        self.threshold_a = threshold_a  # Edge prediction threshold

        self.attack_node_number = int(self.num_nodes * self.attack_node_fraction)
        self.graph_data = self.graph_data.to(self.device)
        self.graph = self.graph_data
        self.features = self.graph.ndata['feat']

        # Initialize edge predictor and surrogate model
        self.edge_predictor = DGLEdgePredictor(
            self.num_features, hidden_dim, self.num_classes, self.device
        ).to(self.device)

        self.surrogate_model = DGLSurrogateModel(
            self.num_features, self.num_classes
        ).to(self.device)

        # net
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)

        # Optimizers
        self.optimizer_edge = optim.Adam(self.edge_predictor.parameters(), lr=0.01, weight_decay=5e-4)
        self.optimizer_surrogate = optim.Adam(self.surrogate_model.parameters(), lr=0.01, weight_decay=5e-4)

        print(f"Initialized attack on {dataset.dataset_name} dataset")
        print(f"Nodes: {self.num_nodes}, Features: {self.num_features}, Classes: {self.num_classes}")
        print(f"Attack nodes: {self.attack_node_number} ({attack_node_fraction:.1%})")

    def simulate_target_model_queries(self, query_nodes, error_rate=0.15):
        """Simulate target model queries with a certain proportion of incorrect labels"""
        self.net1.eval()
        with torch.no_grad():
            logits = self.net1(self.graph, self.features)
            predictions = F.log_softmax(logits, dim=1).argmax(dim=1)

        # Get predicted labels for query nodes
        predicted_labels = predictions[query_nodes].clone()

        # Introduce incorrect labels
        num_errors = int(len(predicted_labels) * error_rate)
        if num_errors > 0:
            error_indices = random.sample(range(len(predicted_labels)), num_errors)
            for idx in error_indices:
                # Randomly assign incorrect labels
                wrong_label = random.randint(0, self.num_classes - 1)
                predicted_labels[idx] = wrong_label

        return predicted_labels

    def compute_cosine_similarity(self, features):
        """Compute cosine similarity of node features"""
        features_np = features.cpu().detach().numpy()
        similarity_matrix = cosine_similarity(features_np)
        return torch.tensor(similarity_matrix, dtype=torch.float32, device=self.device)

    def generate_candidate_edges(self, labeled_nodes, unlabeled_nodes):
        """Generate candidate edge set"""
        similarity_matrix = self.compute_cosine_similarity(self.features)
        candidate_edges = []

        for u_node in unlabeled_nodes:
            for l_node in labeled_nodes:
                if similarity_matrix[u_node, l_node] > self.threshold_s:
                    candidate_edges.append([u_node, l_node])

        print(f"Generated {len(candidate_edges)} candidate edges based on cosine similarity")
        return candidate_edges

    def train_edge_predictor(self, labeled_nodes, predicted_labels, epochs=100):
        """Train edge prediction model"""
        print("Training edge predictor...")
        self.edge_predictor.train()

        # Create training labels - only queried nodes have labels, others are -1
        train_labels = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        train_labels[labeled_nodes] = predicted_labels

        for epoch in range(epochs):
            self.optimizer_edge.zero_grad()

            # Forward pass
            node_embeddings, node_logits = self.edge_predictor(self.graph, self.features)

            # Node classification loss (only for labeled nodes)
            labeled_mask = train_labels != -1
            if labeled_mask.sum() > 0:
                node_loss = F.cross_entropy(node_logits[labeled_mask], train_labels[labeled_mask])
            else:
                node_loss = torch.tensor(0.0, device=self.device)

            # Edge prediction loss
            src_nodes, dst_nodes = self.graph.edges()
            positive_pairs = list(zip(src_nodes.cpu().numpy(), dst_nodes.cpu().numpy()))

            # Positive samples
            pos_edge_probs = self.edge_predictor.predict_edges(node_embeddings, positive_pairs)
            pos_loss = -torch.log(pos_edge_probs + 1e-15).mean()

            # Negative samples
            negative_pairs = []
            num_neg_samples = min(len(positive_pairs), 1000)  # Limit negative sample size
            for _ in range(num_neg_samples):
                src = random.randint(0, self.num_nodes - 1)
                dst = random.randint(0, self.num_nodes - 1)
                if src != dst and not self.graph_data.has_edges_between(src, dst):
                    negative_pairs.append([src, dst])

            if negative_pairs:
                neg_edge_probs = self.edge_predictor.predict_edges(node_embeddings, negative_pairs)
                neg_loss = -torch.log(1 - neg_edge_probs + 1e-15).mean()
            else:
                neg_loss = torch.tensor(0.0, device=self.device)

            # Total loss
            total_loss = node_loss + 0.5 * (pos_loss + neg_loss)

            total_loss.backward()
            self.optimizer_edge.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Total Loss: {total_loss.item():.4f}, "
                      f"Node Loss: {node_loss.item():.4f}, Edge Loss: {(pos_loss + neg_loss).item():.4f}")

    def add_potential_edges(self, candidate_edges, labeled_nodes):
        """Add potential edges based on edge prediction results"""
        if not candidate_edges:
            return self.graph

        print("Predicting edge weights and adding potential edges...")

        self.edge_predictor.eval()
        with torch.no_grad():
            node_embeddings, _ = self.edge_predictor(self.graph, self.features)
            edge_probs = self.edge_predictor.predict_edges(node_embeddings, candidate_edges)

        # Select edges with probability above threshold
        selected_edges = []
        for i, (src, dst) in enumerate(candidate_edges):
            if edge_probs[i] > self.threshold_a:
                selected_edges.extend([(src, dst), (dst, src)])  # Undirected graph

        print(f"Selected {len(selected_edges) // 2} potential edges to add")

        if selected_edges:
            # Create new graph and add edges
            enhanced_graph = dgl.add_edges(
                self.graph,
                [e[0] for e in selected_edges],
                [e[1] for e in selected_edges]
            )
            return enhanced_graph
        else:
            return self.graph

    def train_surrogate_model(self, enhanced_graph, labeled_nodes, predicted_labels, epochs=200):
        """Train surrogate model"""
        print("Training surrogate model...")
        self.surrogate_model.train()

        # Create training labels
        train_labels = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        train_labels[labeled_nodes] = predicted_labels
        labeled_mask = train_labels != -1

        for epoch in range(epochs):
            self.optimizer_surrogate.zero_grad()

            # Forward pass
            logits = self.surrogate_model(enhanced_graph, self.features)

            if labeled_mask.sum() > 0:
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[labeled_mask], train_labels[labeled_mask])

                loss.backward()
                self.optimizer_surrogate.step()

                if epoch % 50 == 0:
                    print(f"Surrogate Model Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    def evaluate_attack(self, enhanced_graph):
        """Evaluate attack performance"""
        print("\nEvaluating attack performance...")

        # Evaluate surrogate model
        self.surrogate_model.eval()
        with torch.no_grad():
            surrogate_logits = self.surrogate_model(enhanced_graph, self.features)
            surrogate_pred = F.log_softmax(surrogate_logits, dim=1).argmax(dim=1)

            # Calculate accuracy on test set
            test_acc = (surrogate_pred[self.graph_data.ndata['test_mask']] == self.graph_data.ndata['label'][
                self.graph_data.ndata['test_mask']]).float().mean()

        # Evaluate fidelity with target model
        self.net1.eval()
        with torch.no_grad():
            target_logits = self.net1(self.graph, self.features)
            target_pred = F.log_softmax(target_logits, dim=1).argmax(dim=1)

            # Calculate fidelity (consistency on test set)
            fidelity = (surrogate_pred[self.graph_data.ndata['test_mask']] == target_pred[
                self.graph_data.ndata['test_mask']]).float().mean()

        return test_acc.item(), fidelity.item(), surrogate_pred

    def attack(self):
        """Execute model extraction attack"""
        print("=" * 60)
        print("Starting GNN Model Extraction Attack")
        print("=" * 60)

        # Step 1: Randomly select query nodes
        all_nodes = list(range(self.num_nodes))
        labeled_nodes = random.sample(all_nodes, self.attack_node_number)
        unlabeled_nodes = [n for n in all_nodes if n not in labeled_nodes]

        print(f"Selected {len(labeled_nodes)} nodes for querying")

        # Step 2: Simulate target model queries
        predicted_labels = self.simulate_target_model_queries(labeled_nodes)
        print(f"Simulated target model queries with ~15% error rate")

        # Step 3: Generate candidate edges
        candidate_edges = self.generate_candidate_edges(labeled_nodes, unlabeled_nodes)

        # Step 4: Train edge prediction model
        self.train_edge_predictor(labeled_nodes, predicted_labels)

        # Step 5: Add potential edges
        enhanced_graph = self.add_potential_edges(candidate_edges, labeled_nodes)

        original_edges = self.graph_data.num_edges()
        enhanced_edges = enhanced_graph.num_edges()
        print(f"Enhanced graph: {original_edges} -> {enhanced_edges} edges (+{enhanced_edges - original_edges})")

        # Step 6: Train surrogate model
        self.train_surrogate_model(enhanced_graph, labeled_nodes, predicted_labels)

        # Step 7: Evaluate attack performance
        test_accuracy, fidelity, predictions = self.evaluate_attack(enhanced_graph)

        print("=" * 60)
        print("Attack Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Fidelity (vs Target Model): {fidelity:.4f}")
        print("=" * 60)

        return {
            'test_accuracy': test_accuracy,
            'fidelity': fidelity,
            'enhanced_graph': enhanced_graph,
            'predictions': predictions,
            'labeled_nodes': labeled_nodes,
            'predicted_labels': predicted_labels
        }
