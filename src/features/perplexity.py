"""
Advanced Multi-Model Perplexity Extractor

Extracts 8 perplexity-based features using multiple language models:
1. Qwen2.5-Coder perplexity
2. StarCoder2 perplexity
3. Windowed perplexity mean
4. Windowed perplexity std
5. Windowed perplexity range
6. Conditional line perplexity
7. Cross-model perplexity agreement
8. Token-level perplexity variance

AI-generated code typically has:
- Lower overall perplexity (more predictable)
- Lower variance in perplexity (more uniform)
- Higher cross-model agreement
- Different windowed patterns
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiModelPerplexityExtractor:
    """Extract perplexity features using multiple language models."""
    
    def __init__(
        self,
        models: List[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_8bit: bool = True,
        window_size: int = 50
    ):
        """
        Initialize perplexity extractors for multiple models.
        
        Args:
            models: List of model names to use for perplexity computation
            device: Device to run models on
            use_8bit: Whether to use 8-bit quantization
            window_size: Window size for windowed perplexity (in tokens)
        """
        if models is None:
            # Default: use lightweight models for efficiency
            models = [
                "Qwen/Qwen2.5-Coder-1.5B",  # Qwen coder model
                "bigcode/starcoder2-3b"      # StarCoder2
            ]
        
        self.device = device
        self.window_size = window_size
        self.models = {}
        self.tokenizers = {}
        
        # Load models with 8-bit quantization if requested
        quantization_config = None
        if use_8bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        # Use bfloat16 on CUDA for better numerical stability, float32 on CPU
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using dtype: {dtype}")
        
        print("Loading perplexity models...")
        for model_name in models:
            try:
                print(f"  Loading {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if use_8bit else None,
                    trust_remote_code=True,
                    torch_dtype=dtype
                )
                
                if not use_8bit:
                    model = model.to(device)
                
                model.eval()
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                print(f"  ✓ Loaded {model_name}")
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
        
        if not self.models:
            raise RuntimeError("Failed to load any perplexity models")
    
    def compute_perplexity(
        self,
        code: str,
        model_name: str,
        max_length: int = 512
    ) -> float:
        """
        Compute perplexity for code using a specific model.
        
        Args:
            code: Source code string
            model_name: Name of the model to use
            max_length: Maximum sequence length
            
        Returns:
            Perplexity score (lower = more predictable/AI-like)
        """
        if model_name not in self.models:
            return float('inf')
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize code
            inputs = tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            
            # Compute loss
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            print(f"Error computing perplexity with {model_name}: {e}")
            return float('inf')
    
    def compute_windowed_perplexity(
        self,
        code: str,
        model_name: str
    ) -> Dict[str, float]:
        """
        Compute perplexity over sliding windows of code.
        
        Returns:
            Dictionary with 'mean', 'std', 'range' of windowed perplexities
        """
        if model_name not in self.models:
            return {"mean": float('inf'), "std": 0.0, "range": 0.0}
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize full code
            tokens = tokenizer.encode(code, add_special_tokens=False)
            
            if len(tokens) < self.window_size:
                # Code too short, use full perplexity
                ppl = self.compute_perplexity(code, model_name)
                return {"mean": ppl, "std": 0.0, "range": 0.0}
            
            # Compute perplexity for each window
            window_ppls = []
            stride = max(1, self.window_size // 2)  # 50% overlap
            
            for start_idx in range(0, len(tokens) - self.window_size + 1, stride):
                window_tokens = tokens[start_idx:start_idx + self.window_size]
                window_text = tokenizer.decode(window_tokens)
                
                ppl = self.compute_perplexity(window_text, model_name, max_length=self.window_size)
                if ppl != float('inf'):
                    window_ppls.append(ppl)
            
            if not window_ppls:
                return {"mean": float('inf'), "std": 0.0, "range": 0.0}
            
            window_ppls = np.array(window_ppls)
            
            return {
                "mean": float(np.mean(window_ppls)),
                "std": float(np.std(window_ppls)),
                "range": float(np.max(window_ppls) - np.min(window_ppls))
            }
            
        except Exception as e:
            print(f"Error computing windowed perplexity: {e}")
            return {"mean": float('inf'), "std": 0.0, "range": 0.0}
    
    def compute_conditional_perplexity(
        self,
        code: str,
        model_name: str
    ) -> float:
        """
        Compute average line-by-line conditional perplexity.
        
        This measures how predictable each line is given previous lines.
        AI code tends to have lower conditional perplexity.
        """
        if model_name not in self.models:
            return float('inf')
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            lines = code.split('\n')
            if len(lines) < 2:
                return self.compute_perplexity(code, model_name)
            
            conditional_ppls = []
            
            for i in range(1, min(len(lines), 20)):  # Limit to first 20 lines
                context = '\n'.join(lines[:i])
                target_line = lines[i]
                
                if len(target_line.strip()) == 0:
                    continue
                
                full_text = context + '\n' + target_line
                
                # Tokenize
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                labels = inputs["input_ids"].clone()
                
                # Mask context tokens (only compute loss on target line)
                context_inputs = tokenizer(context + '\n', return_tensors="pt", truncation=True, max_length=512)
                context_length = context_inputs["input_ids"].shape[1]
                labels[:, :context_length] = -100  # Ignore context in loss
                
                input_ids = inputs["input_ids"].to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    conditional_ppls.append(torch.exp(loss).item())
            
            if not conditional_ppls:
                return self.compute_perplexity(code, model_name)
            
            return float(np.mean(conditional_ppls))
            
        except Exception as e:
            print(f"Error computing conditional perplexity: {e}")
            return float('inf')
    
    def compute_cross_model_agreement(
        self,
        code: str
    ) -> float:
        """
        Compute agreement between multiple models' perplexities.
        
        High agreement (low variance) suggests AI-generated code,
        as different models trained on similar data will agree.
        
        Returns:
            Normalized agreement score [0-1], where higher = more agreement
        """
        ppls = []
        
        for model_name in self.models.keys():
            ppl = self.compute_perplexity(code, model_name)
            if ppl != float('inf'):
                ppls.append(ppl)
        
        if len(ppls) < 2:
            return 0.0
        
        ppls = np.array(ppls)
        
        # Compute coefficient of variation (std / mean)
        # Lower CV = higher agreement
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        
        if mean_ppl == 0:
            return 0.0
        
        cv = std_ppl / mean_ppl
        
        # Normalize to [0, 1] where 1 = high agreement
        # Typical CV for code is 0.1 to 1.0
        agreement = np.exp(-cv)
        
        return float(agreement)
    
    def compute_token_perplexity_variance(
        self,
        code: str,
        model_name: str,
        max_length: int = 256
    ) -> float:
        """
        Compute variance of per-token perplexities.
        
        AI code tends to have lower variance (more uniform predictability).
        """
        if model_name not in self.models:
            return 0.0
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            inputs = tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                
                # Get per-token losses
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                # Compute per-token loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Compute variance
                variance = token_losses.var().item()
            
            return float(variance)
            
        except Exception as e:
            print(f"Error computing token perplexity variance: {e}")
            return 0.0
    
    def extract(self, code: str, language: str = "python") -> List[float]:
        """
        Extract all 8 perplexity features.
        
        Features:
        1. Qwen perplexity
        2. StarCoder perplexity
        3. Windowed perplexity mean
        4. Windowed perplexity std
        5. Windowed perplexity range
        6. Conditional line perplexity
        7. Cross-model agreement
        8. Token perplexity variance
        
        Returns:
            List of 8 normalized features
        """
        if not code or len(code.strip()) == 0:
            return [0.0] * 8
        
        features = []
        model_names = list(self.models.keys())
        
        try:
            # Feature 1-2: Basic perplexity from each model
            for model_name in model_names[:2]:  # Use first 2 models
                ppl = self.compute_perplexity(code, model_name)
                # Normalize to [0, 1]: typical perplexity is 2-50
                normalized_ppl = 1.0 - min(1.0, ppl / 50.0)
                features.append(normalized_ppl)
            
            # Pad if less than 2 models
            while len(features) < 2:
                features.append(0.0)
            
            # Features 3-5: Windowed perplexity statistics
            windowed = self.compute_windowed_perplexity(code, model_names[0])
            features.append(1.0 - min(1.0, windowed["mean"] / 50.0))
            features.append(min(1.0, windowed["std"] / 10.0))
            features.append(min(1.0, windowed["range"] / 20.0))
            
            # Feature 6: Conditional line perplexity
            conditional_ppl = self.compute_conditional_perplexity(code, model_names[0])
            features.append(1.0 - min(1.0, conditional_ppl / 50.0))
            
            # Feature 7: Cross-model agreement
            agreement = self.compute_cross_model_agreement(code)
            features.append(agreement)
            
            # Feature 8: Token-level variance
            variance = self.compute_token_perplexity_variance(code, model_names[0])
            features.append(min(1.0, variance / 5.0))
            
        except Exception as e:
            print(f"Error extracting perplexity features: {e}")
            # Return default features on error
            features = [0.0] * 8
        
        # Ensure we have exactly 8 features
        return features[:8] + [0.0] * max(0, 8 - len(features))


if __name__ == "__main__":
    # Test the perplexity extractor
    print("Testing Multi-Model Perplexity Extractor")
    print("=" * 50)
    
    # Sample codes
    human_code = """
def calculate_sum(lst):
    # quick sum
    total = 0
    for x in lst:
        total += x
    return total
"""
    
    ai_code = """
def calculate_sum(numbers: List[int]) -> int:
    '''
    Calculate the sum of all numbers in the input list.
    
    Args:
        numbers: A list of integers to be summed
        
    Returns:
        The total sum of all integers in the list
        
    Example:
        >>> calculate_sum([1, 2, 3])
        6
    '''
    total_sum = 0
    for number in numbers:
        total_sum = total_sum + number
    return total_sum
"""
    
    print("\nInitializing extractor...")
    extractor = MultiModelPerplexityExtractor(use_8bit=False)  # Use full precision for quick testing
    
    print("\nExtracting features from human code:")
    human_features = extractor.extract(human_code)
    print(f"Features: {[f'{f:.3f}' for f in human_features]}")
    
    print("\nExtracting features from AI code:")
    ai_features = extractor.extract(ai_code)
    print(f"Features: {[f'{f:.3f}' for f in ai_features]}")
    
    print("\nFeature comparison (Human vs AI):")
    feature_names = [
        "Qwen PPL",
        "StarCoder PPL",
        "Window Mean",
        "Window Std",
        "Window Range",
        "Conditional PPL",
        "Cross-Model Agr",
        "Token Variance"
    ]
    
    for name, h_feat, a_feat in zip(feature_names, human_features, ai_features):
        diff = a_feat - h_feat
        print(f"  {name:20s}: Human={h_feat:.3f}, AI={a_feat:.3f}, Diff={diff:+.3f}")
    
    print("\n" + "=" * 50)
    print("Expected AI patterns:")
    print("  - Higher normalized PPL (lower raw perplexity)")
    print("  - Lower window variance")
    print("  - Higher cross-model agreement")
    print("  - Lower token variance")
