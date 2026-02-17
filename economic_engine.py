import torch
import torch.nn as nn
import hashlib
import uuid
import argparse

# --- 1. Your Data Structure (Preserved) ---
def create_node(content, node_type, parent=None):
    """Create a logical snapshot node."""
    return {
        "id": str(uuid.uuid4()),        # unique node ID
        "type": node_type,              # book / chapter / paragraph / sentence / concept
        "content": content,             # raw or summarized text
        "parent": parent,               # parent node ID
        "children": []                  # child node IDs
    }

# --- 2. The Recursive Economic Engine ---
class RecursiveEconomicEngine(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.dim = embedding_dim

        # A. The "Raw Material" Extractor
        # deterministic hashing to turn text strings into initial vectors
        self.seed_generator = torch.Generator()

        # B. The "Means of Production" (Transformation Matrices)
        # Different logic for different levels of abstraction
        self.transforms = nn.ModuleDict({
            'concept': nn.Linear(embedding_dim, embedding_dim),
            'paragraph': nn.Linear(embedding_dim, embedding_dim),
            'chapter': nn.Linear(embedding_dim, embedding_dim),
            'book_section': nn.Linear(embedding_dim, embedding_dim),
            'book': nn.Linear(embedding_dim, embedding_dim)
        })

        # C. The "Invisible Hand" (Aggregation Logic)
        # How do we combine many children into one parent?
        self.aggregator = nn.Linear(embedding_dim * 2, embedding_dim)

    def _text_to_vector(self, text):
        """Converts raw text content into a 'Natural Resource' vector."""
        # Create a deterministic seed from the text content
        # (The value is inherent in the object, not assigned randomly)
        hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        self.seed_generator.manual_seed(hash_val)
        return torch.randn(self.dim, generator=self.seed_generator)

    def forward(self, node):
        """
        Recursive function to calculate the 'Economic Value' of a node.
        Logic: Value = Transform( Self_Content + Aggregate(Children_Values) )
        """
        # 1. Extract value from the node's own content (Self-Labor)
        self_vector = self._text_to_vector(node['content'])

        # 2. Extract value from Children (Accumulated Capital)
        if not node['children']:
            # Base Case: Leaf node (Concept)
            child_aggregate = torch.zeros(self.dim)
        else:
            # Recursive Step: Gather wealth from all children
            child_vectors = [self.forward(child) for child in node['children']]

            # Stack and pool the children (Mean Pooling = 'Average Market Value')
            child_stack = torch.stack(child_vectors)
            child_aggregate = torch.mean(child_stack, dim=0)

        # 3. Production Process
        # Combine Self-Input with Child-Input
        combined = torch.cat([self_vector, child_aggregate], dim=0)

        # Apply the logic of the specific hierarchy level (Paragraph vs Chapter logic)
        # We perform a 'compression' from 2xDim to 1xDim
        processed = torch.tanh(self.aggregator(combined))

        # Apply specific transformation based on node type (e.g., 'paragraph')
        if node['type'] in self.transforms:
            final_output = self.transforms[node['type']](processed)
        else:
            final_output = processed

        return final_output

# --- 3. Execution Simulation ---

def run_simulation():
    # Example snapshot of a small part of the book
    book_snapshot = create_node("The Wealth of Nations", "book")

    # Book 1
    book1 = create_node("Book 1: Of the Causes of Improvement in the productive Powers of Labour", "book_section", parent=book_snapshot["id"])
    book_snapshot["children"].append(book1)

    # Chapter 1
    chapter1 = create_node("Chapter 1: Of the Division of Labour", "chapter", parent=book1["id"])
    book1["children"].append(chapter1)

    # Paragraph 1
    para1 = create_node(
        "The greatest improvements in the productive powers of labour, and the greater part of the skill, dexterity, and judgement with which it is anywhere directed, seem to have been the effects of the division of labour.",
        "paragraph",
        parent=chapter1["id"]
    )
    chapter1["children"].append(para1)

    # Sentence-level concepts (concept snapshot)
    concept1 = create_node("Division of labour increases productivity.", "concept", parent=para1["id"])
    para1["children"].append(concept1)

    concept2 = create_node("Skill and judgement are enhanced by specialization.", "concept", parent=para1["id"])
    para1["children"].append(concept2)

    # --- Run the Engine ---
    embedding_dim = 128
    engine = RecursiveEconomicEngine(embedding_dim=embedding_dim)

    # Calculate the vector for the Root Node.
    # This triggers a chain reaction down to the concepts and bubbles back up.
    final_book_vector = engine(book_snapshot)

    # Calculate vector for a sub-component (e.g., just the paragraph)
    paragraph_vector = engine(para1)

    print("--- Hierarchical Value Calculation ---")
    print(f"Structure Depth: 5 levels")
    print(f"Paragraph Vector (First 5 dims): {paragraph_vector.detach().numpy()[:5]}")
    print(f"Total Book Vector (First 5 dims): {final_book_vector.detach().numpy()[:5]}")

    # Calculate Semantic Similarity (Dot Product)
    # How much of the 'Paragraph' is contained in the final 'Book'?
    similarity = torch.dot(paragraph_vector, final_book_vector) / (torch.norm(paragraph_vector) * torch.norm(final_book_vector))
    print(f"Semantic Alignment (Paragraph vs Book): {similarity.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive Economic Engine Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the engine simulation")
    args = parser.parse_args()

    # Run simulation if requested or by default
    if args.simulate or not any(vars(args).values()):
        run_simulation()
