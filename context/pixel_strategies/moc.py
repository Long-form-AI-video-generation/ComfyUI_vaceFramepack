import torch
from .base_strategy import ContextStrategy
from ..cache_manager import CacheManager
from ...utils.tensor_utils import cosine_similarity_matrix, select_and_sort, perform_soft_nms

class MoCStrategy(ContextStrategy):
    def __init__(self):
        self.cache = CacheManager.get_instance()
        
    def select(self, history_images: torch.Tensor, context_size: int, **kwargs) -> torch.Tensor:
        """
        Industry-Grade MoC Implementation.
        Features: Hybrid Continuity, Multi-Modal Fusion, Dimension Safety, Soft-NMS.
        """
        if history_images is None or history_images.shape[0] == 0:
            return torch.empty((0, 512, 512, 3)) 

    
        clip_vision = kwargs.get('clip_vision')
        text_encoder = kwargs.get('text_encoder')
        current_prompt = kwargs.get('current_prompt')
        
        contiguous_size = kwargs.get('contiguous_size', 4)
        text_weight = kwargs.get('text_weight', 0.0)
        diversity_radius = kwargs.get('diversity_radius', 0)
        similarity_threshold = kwargs.get('similarity_threshold', 0.0)
        
        total_frames = history_images.shape[0]
        
    
        if total_frames <= context_size and similarity_threshold <= 0:
            return history_images
            
    
        actual_contiguous_size = min(contiguous_size, total_frames)
        recent_block_start = total_frames - actual_contiguous_size
        recent_block = history_images[recent_block_start:]
        
        slots_remaining = context_size - actual_contiguous_size
        if slots_remaining <= 0 or recent_block_start <= 0:
            return recent_block
            
    
        if clip_vision is None:
            return history_images[-context_size:]

        def _encode_wrapper(imgs):
            return clip_vision.encode_image(imgs).float().reshape(imgs.shape[0], -1)

    
        full_embeds = self.cache.get_embeddings(history_images, _encode_wrapper, clip_vision)
        if full_embeds is None:
             return history_images[-context_size:]

    
    
        visual_query = full_embeds[-1].unsqueeze(0)
        search_pool_embeds = full_embeds[:recent_block_start]
        
    
        if search_pool_embeds.shape[0] == 0:
            return recent_block

        visual_scores = cosine_similarity_matrix(visual_query, search_pool_embeds)
        final_scores = visual_scores
        
    
        if text_encoder is not None and isinstance(current_prompt, str) and text_weight > 0.0 and len(current_prompt) > 0:
            try:
                pooled_text = None
            
                if hasattr(text_encoder, 'encode_text'):
                    pooled_text = text_encoder.encode_text(current_prompt)
                elif hasattr(text_encoder, 'encode'):
                    pooled_text = text_encoder.encode(current_prompt)
                elif hasattr(text_encoder, 'encode_from_tokens') and hasattr(text_encoder, 'tokenize'):
                    tokens = text_encoder.tokenize(current_prompt)
                    _, pooled_text = text_encoder.encode_from_tokens(tokens, return_pooled=True)
                elif callable(text_encoder):
                
                    pooled_text = text_encoder(current_prompt)

                if pooled_text is None:
                    raise RuntimeError('Text encoder did not return embeddings')

            
                if not isinstance(pooled_text, torch.Tensor):
                    try:
                        pooled_text = torch.as_tensor(pooled_text)
                    except Exception:
                        raise RuntimeError('Unable to convert pooled_text to torch.Tensor')

            
                pooled_text = pooled_text.to(device=search_pool_embeds.device, dtype=search_pool_embeds.dtype)

            
                if pooled_text.dim() == 1:
                    pooled_text = pooled_text.unsqueeze(0)
                elif pooled_text.dim() == 2 and pooled_text.shape[0] != 1:
                
                    pooled_text = pooled_text.mean(dim=0, keepdim=True)

                if pooled_text.dim() != 2:
                    raise RuntimeError('Text encoder returned tensor with unsupported shape')

            
                if pooled_text.shape[-1] != search_pool_embeds.shape[-1]:
                    print(f"[WanVideo Context] Dimension Mismatch! Vision:{search_pool_embeds.shape[-1]}, Text:{pooled_text.shape[-1]}. Disabling Text MoC.")
                else:
                    text_scores = cosine_similarity_matrix(pooled_text, search_pool_embeds)
                    final_scores = (1.0 - text_weight) * visual_scores + (text_weight) * text_scores
            except Exception as e:
                print(f"[WanVideo Context] Text Encode Error: {e}. Falling back to visual-only.")

    
        selected_indices = perform_soft_nms(
            scores=final_scores,
            k=slots_remaining,
            radius=diversity_radius,
            threshold=similarity_threshold
        )
        
    
        if len(selected_indices) == 0:
            return recent_block
            
        selected_distant_frames = select_and_sort(history_images, selected_indices)
        
    
        return torch.cat([selected_distant_frames, recent_block], dim=0)
