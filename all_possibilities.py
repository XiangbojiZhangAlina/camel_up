"""
generate_4d_tensor.py

This script generates the 4D integer tensor representing all possible permutations
of dice order (6P5=720) and all possible face combinations (6^5=7776). The resulting
tensor has shape (720, 7776, 5, 2), where
    tensor[i,j,k,0] = dice ID (1–6) for the k-th roll in the i-th order
    tensor[i,j,k,1] = face value   (1–6) for the k-th roll in the j-th face combo

The tensor is saved in compressed NPZ format to save space.
"""
import itertools
import numpy as np

# Define dice IDs and face values
dice_ids = np.arange(1, 7, dtype=np.uint8)      # 1–6
face_vals = np.arange(1, 7, dtype=np.uint8)      # 1–6

# Generate all ordered permutations of dice (6P5 = 720 x 5)
permutations = np.array(list(itertools.permutations(dice_ids, 5)), dtype=np.uint8)  # shape (720,5)

# Generate all face combinations (6^5 = 7776 x 5)
face_combos = np.array(list(itertools.product(face_vals, repeat=5)), dtype=np.uint8)  # shape (7776,5)

# Use broadcasting to build the 4D tensor (720, 7776, 5, 2)
# dice_layer expands to (720, 7776, 5)
dice_layer = permutations[:, np.newaxis, :]          # shape (720,1,5)
dice_layer = np.repeat(dice_layer, face_combos.shape[0], axis=1)
# face_layer expands to (720, 7776, 5)
face_layer = face_combos[np.newaxis, :, :]           # shape (1,7776,5)
face_layer = np.repeat(face_layer, permutations.shape[0], axis=0)

# Stack along last axis to get shape (720, 7776, 5, 2)
tensor_4d = np.stack([dice_layer, face_layer], axis=-1)  # dtype=uint8

# Save compressed
output_path = "permutation_face_tensor.npz"
np.savez_compressed(output_path, tensor=tensor_4d)
print(f"Generated 4D tensor of shape {tensor_4d.shape} and saved to {output_path}")
