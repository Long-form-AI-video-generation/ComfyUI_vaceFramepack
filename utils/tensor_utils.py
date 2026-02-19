import torch

def safe_slice(tensor: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
    if tensor is None:
        return torch.empty(0)
    
    total_len = tensor.shape[0]
    start = max(0, min(start_idx, total_len))
    end = max(0, min(end_idx, total_len))
    
    if start >= end:
        return torch.empty((0, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        
    return tensor[start:end]

def cosine_similarity_matrix(query_embedding: torch.Tensor, key_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between a single query (1, D) and keys (N, D).
    """
    eps = 1e-8
    query_norm = query_embedding / (query_embedding.norm(dim=1, keepdim=True) + eps)
    keys_norm = key_embeddings / (key_embeddings.norm(dim=1, keepdim=True) + eps)
    return torch.mm(query_norm, keys_norm.transpose(0, 1)).squeeze(0)

def select_and_sort(history_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Selects frames by index and strictly sorts them to maintain temporal causality.
    """
    if isinstance(indices, list):
        indices = torch.tensor(indices, device=history_tensor.device)
    
    sorted_indices, _ = torch.sort(indices)
    return torch.index_select(history_tensor, 0, sorted_indices)

def perform_soft_nms(scores: torch.Tensor, k: int, radius: int, threshold: float = 0.0) -> torch.Tensor:
    """
    Selects indices using Soft Non-Maximum Suppression to enforce diversity.
    
    Args:
        scores: (N,) Tensor of similarity scores (-1 to 1).
        k: Maximum number of frames to select.
        radius: Suppression window size (temporal distance).
        threshold: Minimum score to accept.
        
    Returns:
        Tensor of selected indices.
    """

    work_scores = scores.clone().float() 
    selected_indices = []

    decay_factor = 0.5 

    for _ in range(k):
        best_idx = int(torch.argmax(work_scores).item())
        score_val = float(work_scores[best_idx].item())
        if score_val < threshold:
            break
        
        selected_indices.append(best_idx)

        work_scores[best_idx] = -float('inf')
        
        if radius > 0:
            start_r = int(max(0, best_idx - radius))
            end_r = int(min(work_scores.shape[0], best_idx + radius + 1))
            mask = work_scores[start_r:end_r] > 0
            work_scores[start_r:end_r][mask] *= decay_factor

    return torch.tensor(selected_indices, dtype=torch.long, device=scores.device)
