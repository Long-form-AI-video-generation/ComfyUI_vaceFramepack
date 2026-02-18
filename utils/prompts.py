"""
Prompt handling utilities for FramePack video generation.
Contains prompt parsing, weight extraction, and text encoding.
"""

import re
import torch

from comfy import model_management as mm


class PromptHandler:
    """Handles prompt parsing and encoding"""
    
    @staticmethod
    def parse_multi_prompts(multi_prompts, num_sections):
        """
        Parse multi-line prompts and assign them to sections.
        Each line represents a prompt for a section.
        If fewer prompts than sections, the last prompt is repeated.
        """
        # Split by newline and filter empty lines
        prompts = [p.strip() for p in multi_prompts.split('\n') if p.strip()]
        
        if not prompts:
            raise ValueError("No prompts provided in multi_prompts")
        
        # Assign prompts to sections
        section_prompts = []
        for section in range(num_sections):
            if section < len(prompts):
                section_prompts.append(prompts[section])
            else:
                # Repeat the last prompt for remaining sections
                section_prompts.append(prompts[-1])
        
        print(f"Parsed {len(prompts)} unique prompts for {num_sections} sections")
        for i, prompt in enumerate(section_prompts):
            print(f"  Section {i}: {prompt[:50]}...")
        
        return section_prompts
    
    @staticmethod
    def parse_prompt_weights(prompt):
        """
        Parse prompt weights in the format (text:weight).
        Returns cleaned prompt and weight dictionary.
        """
        weights = {}
        cleaned_prompt = prompt
        
        # Pattern to find (text:weight) format
        pattern = r'\(([^:)]+):([0-9.]+)\)'
        matches = re.findall(pattern, prompt)
        
        for text, weight_str in matches:
            try:
                weight = float(weight_str)
                weights[text.strip()] = weight
                # Remove the weight notation from the prompt
                cleaned_prompt = cleaned_prompt.replace(f"({text}:{weight_str})", text)
            except ValueError:
                print(f"Invalid weight value: {weight_str}")
        
        return cleaned_prompt.strip(), weights
    
    @staticmethod
    def encode_prompt_for_section(prompt, negative_prompt, text_encoder, device):
        """
        Encode a single prompt for a section using the WAN Video text encoder.
        Supports weighted prompts using (text:weight) syntax.
        """
        if text_encoder is None:
            raise ValueError("Text encoder is required for encoding prompts")
        
        # Extract the encoder model and dtype
        encoder = text_encoder["model"]
        dtype = text_encoder["dtype"]
        
        # Split positive prompts by '|' and process weights
        positive_prompts_raw = [p.strip() for p in prompt.split('|')]
        positive_prompts = []
        all_weights = []
        
        for p in positive_prompts_raw:
            cleaned_prompt, weights = PromptHandler.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)
        
        # Move encoder to device
        encoder.model.to(device)
        
        try:
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
                # Encode positive and negative prompts
                context = encoder(positive_prompts, device)
                context_null = encoder([negative_prompt if negative_prompt else ""], device)
                
                # Apply weights to embeddings if any were extracted
                for i, weights in enumerate(all_weights):
                    if weights:  # Only apply if weights exist
                        for text, weight in weights.items():
                            print(f"Applying weight {weight} to prompt: {text}")
                            context[i] = context[i] * weight
        finally:
            # Always move encoder back to CPU to free VRAM
            encoder.model.to('cpu')
            mm.soft_empty_cache()
        
        # Create the embedding dictionary with all required fields for WAN Video
        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
        }
        
        return prompt_embeds_dict
