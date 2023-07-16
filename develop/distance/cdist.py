import numpy as np
import timeit

import torch;embeddings = torch.rand((1000,32));atom_embedding = torch.rand((100,32))

setup = "import torch;embeddings = torch.rand((1000,32));atom_embedding = torch.rand((100,32))"

test_code = """
latent_force_distances = torch.cdist(embeddings,atom_embedding,p=2.);
inds = torch.argmin(latent_force_distances,dim=0)
min_vectors = (embeddings[inds]-atom_embedding).abs()
"""
latent_force_distances = torch.cdist(embeddings,atom_embedding,p=2.);
inds = torch.argmin(latent_force_distances,dim=0)
min_vectors = torch.vstack([embeddings[ind]-atom_embedding[i] for i, ind in enumerate(inds)]).abs()
min_vectors2 = (embeddings[inds]-atom_embedding).abs()

assert np.all(np.isclose(min_vectors,min_vectors))

print(
    timeit.timeit(
        test_code,number=1000,setup=setup)
)


