# in this code we test for the decomposition of sentence embeddings in to their component concepts. 

# first import libraries
import numpy as np 
import torch
import json
import csv
import itertools
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

##########################################
# 1. Load Objects, Relations, and Data   #
##########################################

def load_json_data(file_path, key):
    with open(file_path, 'r') as f:
        return json.load(f)[key]

# Load object and relation vocabularies
objects = load_json_data('src/data/objects.json', "objects")
spatial_relations = load_json_data('src/data/relations.json', "spatial_relations")

#print some examples 
print(objects[:5]) 
#extract relations 
relations_pairs = [pair for category in spatial_relations.values() for pair in category]
print(relations_pairs[:5])
# Extract unique relation words from each pair (fixing the unhashable type error)
relation_vocab = sorted(list(set([rel for pair in relations_pairs for rel in pair])))

print(relation_vocab[:5])
# the relation vocab is a list of all the unique relations in the data the rel for 
# For subject and object, we assume they come from the same object vocabulary.
entity_vocab = sorted(objects)

# Build index mappings
subject_to_idx = {word: i for i, word in enumerate(entity_vocab)}
relation_to_idx = {word: i for i, word in enumerate(relation_vocab)}
object_to_idx = {word: i for i, word in enumerate(entity_vocab)}

# print("The subject to index mapping is: ", subject_to_idx)
# print("The relation to index mapping is: ", relation_to_idx)
# print("The object to index mapping is: ", object_to_idx)
# Decide on design matrix dimensions:
# We treat subjects and objects as distinct even if coming from the same vocabulary.
n_subject = len(entity_vocab)
n_relation = len(relation_vocab)
n_object = len(entity_vocab)
total_components = n_subject + n_relation + n_object  # e.g., 50 + 58 + 50 = 158

print(f"Number of subjects: {n_subject}, relations {n_relation}, objects:{n_object} with total components: {total_components}")



##########################################
# 2. Load Sentence Embeddings Data       #
##########################################

DATA_PATH = 'src/data/sentence-embeddings'
MODEL = "Alibaba-NLP/gte-large-en-v1.5"

def get_embeddings_path(model_name):
    return f"{DATA_PATH}/{model_name.replace('/', '_')}.pt"

print("Loading embeddings data...")
data = torch.load(get_embeddings_path(MODEL), weights_only=False)
print(f"Loaded {len(data)} sentence embeddings.")

# B: sentence embedding matrix (each row is an embedding)
B = np.array([item['embedding'] for item in data])
print(f"Sentence embedding matrix B shape: {B.shape}")  # Expect (m, d), e.g., (num_sentences, 1024)



##########################################
# 3. Build Design Matrix A               #
##########################################

def build_design_matrix(data, subject_to_idx, relation_to_idx, object_to_idx, total_components):
    """
    Constructs the design matrix A from sentence metadata.
    Each row represents a sentence, and each column represents a subject, relation, or object.
    """
    A = []
    for item in data:
        row = np.zeros(total_components)
        # Set subject one-hot encoding
        subj_idx = subject_to_idx[item['subject']]
        row[subj_idx] = 1
        # Set relation one-hot encoding (offset by number of subjects)
        rel_idx = relation_to_idx[item['relation']]
        row[n_subject + rel_idx] = 1
        # Set object one-hot encoding (offset by number of subjects + relations)
        obj_idx = object_to_idx[item['object']]
        row[n_subject + n_relation + obj_idx] = 1
        A.append(row)
    return np.array(A)

A = build_design_matrix(data, subject_to_idx, relation_to_idx, object_to_idx, total_components)
print(f"Design matrix A shape: {A.shape}")  # Expected: (m, total_components)

# Print first 5 rows of the design matrix to verify correctness
# print("First 5 rows of design matrix A:")
# print(A[:5])


# Print first 5 rows of A with their corresponding metadata
print("\nVerifying sentence order consistency:")
for i in range(5):
    print(f"Sentence {i}: {data[i]['sentence']}")
    subj = data[i]['subject']
    rel = data[i]['relation']
    obj = data[i]['object']
    subj_idx = subject_to_idx[subj]
    rel_idx = relation_to_idx[rel]
    obj_idx = object_to_idx[obj]
    print(f"Expected indices - Subject: {subj} ({subj_idx}), Relation: {rel} ({rel_idx}), Object: {obj} ({obj_idx})")
    print(f"A[{i}]: {A[i]}")
    print(f"Indices where 1s appear in A[{i}]: {np.where(A[i] == 1)[0]}")
    print("-" * 80)





##########################################
# 4. Solve the Linear System: AX ≈ B     #
##########################################

# Solve for X using the pseudo-inverse (least squares solution)
X = np.linalg.pinv(A) @ B  # X has shape (total_components, d)
print(f"Solved component matrix X shape: {X.shape}")

# Reconstruct the sentence embeddings using the computed X
B_hat = A @ X
reconstruction_loss = np.mean(np.square(B_hat - B))
print(f"Reconstruction loss (MSE) on full dataset: {reconstruction_loss:.2f}")




##########################################
# 5. Statistical Testing                 #
##########################################

def compute_random_losses(A, B, n_iterations=20):
    """
    Shuffle the rows of B to break the correspondence with A,
    solve the linear system, and record the reconstruction loss.
    """
    random_losses = []
    for _ in range(n_iterations):
        B_shuffled = np.copy(B)
        np.random.shuffle(B_shuffled)
        X_rand = np.linalg.pinv(A) @ B_shuffled
        B_hat_rand = A @ X_rand
        loss_rand = np.mean(np.square(B_hat_rand - B_shuffled))
        random_losses.append(loss_rand)
    return random_losses

random_losses = compute_random_losses(A, B, n_iterations=2)
min_random_loss = min(random_losses)
print(f"Minimum reconstruction loss from random permutations: {min_random_loss:.2f}")

if reconstruction_loss < min_random_loss:
    print("Reconstruction loss is significantly lower than random permutations: evidence of compositional structure.")
else:
    print("Reconstruction loss is not significantly lower than random baseline.")

##########################################
# 6. Optimized Leave-One-Out (LOO) Validation  #
##########################################

def leave_one_out_validation(A, B, sample_size=550):
    """
    Optimize LOO by selecting only 2 samples for faster testing.
    This allows for quick validation of correctness.
    """
    similarities = []
    m = A.shape[0]
    sample_indices = np.random.choice(m, min(sample_size, m), replace=False)  # Select 2 samples
    for i in sample_indices:
        print(f"Processing sample {i} / {m}")
        # Create leave-one-out training sets
        A_train = np.delete(A, i, axis=0)
        B_train = np.delete(B, i, axis=0)
        X_loo = np.linalg.pinv(A_train) @ B_train
        # Predict the left-out embedding using its design vector (1 x total_components)
        B_hat_i = A[i:i+1, :] @ X_loo
        sim = cosine_similarity(B[i:i+1, :], B_hat_i)[0][0]
        similarities.append(sim)
        print(f"True embedding: {B[i][:5]}...")  # Print part of the vector for reference
        print(f"Predicted embedding: {B_hat_i[0][:5]}...")  # Partial vector print
        print(f"Cosine similarity: {sim:.4f}")
        print("-" * 80)
    return np.mean(similarities)

loo_avg_similarity = leave_one_out_validation(A, B, sample_size=2)  # Test with only 2 samples
print(f"Approximate average cosine similarity in leave-one-out validation: {loo_avg_similarity:.2f}")
