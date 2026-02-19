import torch
from typing import List, Optional, Callable, Any

class CacheManager:
    _instance: Optional['CacheManager'] = None
    
    def __init__(self):
        self.embedding_cache: List[torch.Tensor] = []
        self._last_encoder_id: int = -1
        
    @classmethod
    def get_instance(cls) -> 'CacheManager':
        if cls._instance is None:
            cls._instance = CacheManager()
        return cls._instance
        
    def clear(self) -> None:
        self.embedding_cache = []
        self._last_encoder_id = -1
        
    def get_embeddings(self, images: torch.Tensor, encoder: Callable[[torch.Tensor], torch.Tensor], encoder_obj: Any, start_index: int = 0) -> Optional[torch.Tensor]:
        """
        Retrieves embeddings, handling cache invalidation if the model changes.
        """

        current_encoder_id = id(encoder_obj)
        if self._last_encoder_id != -1 and current_encoder_id != self._last_encoder_id:
    
            self.clear()
            
        self._last_encoder_id = current_encoder_id

        total_input_frames = images.shape[0]
        cache_len = len(self.embedding_cache)
        

        if start_index == 0 and total_input_frames < cache_len:
            self.clear()
            cache_len = 0


        needed_end = start_index + total_input_frames
        compute_start_global = max(cache_len, start_index)
        tensor_offset = compute_start_global - start_index
        

        if tensor_offset < total_input_frames:
            new_frames = images[tensor_offset:]
            
            with torch.no_grad():
                batch_size = 1
                for i in range(0, new_frames.shape[0], batch_size):
                    batch = new_frames[i : i + batch_size]
                    try:
                        batch_embeds = encoder(batch)
                        for j in range(batch_embeds.shape[0]):
                            self.embedding_cache.append(batch_embeds[j].unsqueeze(0))
                    except Exception as e:
                        print(f"[WanVideo Cache] Error encoding batch: {e}")
                        return None
                            

        if not self.embedding_cache:
            return None
            
        subset = self.embedding_cache[start_index : needed_end]
        return torch.cat(subset, dim=0) if subset else None
