"""
Feature extraction module for AI-generated code detection.

This module provides comprehensive feature extraction including:
- AST features (33): Tree structure, control flow, complexity metrics
- Pattern features (57): LLM-specific coding signatures
- Perplexity features (8): Multi-model predictability analysis
- Stylometric features (12): Code style and formatting patterns

Total: 110 features for Tasks A and C, 145 for Task B with generator-specific features
"""

from .ast_extractor import EnhancedASTExtractor
from .pattern_detector import EnhancedPatternDetector

try:
    from .perplexity import MultiModelPerplexityExtractor
    PERPLEXITY_AVAILABLE = True
except Exception as e:
    print(f"Warning: Perplexity extraction unavailable: {e}")
    PERPLEXITY_AVAILABLE = False

from .stylometric import StylometricExtractor

__all__ = [
    'EnhancedASTExtractor',
    'EnhancedPatternDetector',
    'MultiModelPerplexityExtractor',
    'StylometricExtractor',
    'UnifiedFeatureExtractor',
    'PERPLEXITY_AVAILABLE'
]


import numpy as np
from typing import List, Dict, Optional
import torch


class UnifiedFeatureExtractor:
    """
    Unified interface for extracting all features from code.
    
    Combines AST, pattern, perplexity, and stylometric features
    into a single feature vector.
    """
    
    def __init__(
        self,
        use_perplexity: bool = True,
        perplexity_config: Optional[Dict] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize all feature extractors.
        
        Args:
            use_perplexity: Whether to use perplexity features (requires GPU/CPU time)
            perplexity_config: Configuration for perplexity extractor
            device: Device for perplexity models
        """
        print("Initializing Unified Feature Extractor...")
        
        # Always-available extractors
        self.ast_extractor = EnhancedASTExtractor()
        self.pattern_detector = EnhancedPatternDetector()
        self.stylometric_extractor = StylometricExtractor()
        
        # Optional perplexity extractor
        self.use_perplexity = use_perplexity and PERPLEXITY_AVAILABLE
        self.perplexity_extractor = None
        
        if self.use_perplexity:
            try:
                print("  Loading perplexity models (this may take a moment)...")
                if perplexity_config is None:
                    perplexity_config = {}
                perplexity_config['device'] = device
                self.perplexity_extractor = MultiModelPerplexityExtractor(**perplexity_config)
                print("  [OK] Perplexity extraction enabled")
            except Exception as e:
                print(f"  [FAIL] Failed to load perplexity models: {e}")
                self.use_perplexity = False
        else:
            print("  [X] Perplexity extraction disabled")
        
        print("[OK] Feature extractor ready")
        print(f"  Total features: {self.get_feature_count()}")
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        count = 33 + 57 + 12  # AST + Pattern + Stylometric
        if self.use_perplexity:
            count += 8
        return count
    
    def extract(
        self,
        code: str,
        language: str = "python"
    ) -> np.ndarray:
        """
        Extract all features from code.
        
        Args:
            code: Source code string
            language: Programming language (python, java, cpp, etc.)
            
        Returns:
            NumPy array of shape (feature_count,) with all normalized features
        """
        if not code or len(code.strip()) == 0:
            return np.zeros(self.get_feature_count(), dtype=np.float32)
        
        features = []
        
        # Extract AST features (33 features)
        ast_features = self.ast_extractor.extract(code, language)
        features.extend(ast_features)
        
        # Extract pattern features (57 features)
        pattern_features = self.pattern_detector.extract(code, language)
        features.extend(pattern_features)
        
        # Extract perplexity features (8 features) - optional
        if self.use_perplexity:
            ppl_features = self.perplexity_extractor.extract(code, language)
            features.extend(ppl_features)
        
        # Extract stylometric features (12 features)
        style_features = self.stylometric_extractor.extract(code, language)
        features.extend(style_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch(
        self,
        codes: List[str],
        languages: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract features from multiple code samples.
        
        Args:
            codes: List of source code strings
            languages: Optional list of languages (default: all python)
            
        Returns:
            NumPy array of shape (len(codes), feature_count)
        """
        if languages is None:
            languages = ["python"] * len(codes)
        
        features = []
        for code, lang in zip(codes, languages):
            feat = self.extract(code, lang)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in order."""
        names = []
        
        # AST feature names (33)
        ast_names = [
            "ast_max_depth", "ast_avg_depth", "ast_depth_std", "ast_max_breadth", "ast_tree_balance",
            "ast_node_entropy", "ast_unique_node_ratio", "ast_total_nodes",
            "ast_if_count", "ast_loop_count", "ast_cyclomatic_complexity", "ast_max_nesting", "ast_avg_nesting",
            "ast_func_count", "ast_avg_func_length", "ast_avg_params", "ast_max_func_complexity", "ast_func_with_docstring", "ast_func_with_defaults",
            "ast_class_count", "ast_avg_methods_per_class", "ast_class_with_docstring",
            "ast_docstring_ratio", "ast_comment_density", "ast_avg_docstring_length",
            "ast_type_hint_ratio", "ast_return_annotation_ratio",
            "ast_exception_specificity", "ast_try_coverage",
            "ast_naming_consistency", "ast_avg_identifier_length",
            "ast_import_count", "ast_import_diversity"
        ]
        names.extend(ast_names)
        
        # Pattern feature names (57)
        pattern_names = []
        for family, patterns in [
            ("gpt", ["docstring_style", "markdown_comments", "step_by_step", "verbose_names", 
                     "type_annotations", "comprehensive_error", "educational_tone", "example_blocks"]),
            ("claude", ["numbered_steps", "ultra_verbose", "polite_comments", "careful_error", 
                       "thorough_validation", "methodical_structure"]),
            ("copilot", ["inline_comments", "todo_markers", "short_functions", "pragmatic_style", 
                        "moderate_typing"]),
            ("deepseek", ["fill_in_middle", "defensive_checks", "heavy_typing", "explicit_types", 
                         "conservative_code", "thorough_docstrings"]),
            ("llama", ["long_names", "explicit_logic", "safety_first", "clear_structure", 
                      "minimal_abbreviations", "thorough_comments"]),
            ("starcoder", ["modern_python", "typing_module", "idiomatic", "star_imports", 
                          "efficient_patterns"]),
            ("gemma", ["balanced_verbosity", "practical_comments", "efficient_code", 
                      "standard_patterns"]),
            ("mistral", ["concise_style", "minimal_comments", "performance_focused", 
                        "compact_code"]),
            ("qwen", ["chinese_influence", "thorough_docs", "edge_case_handling", 
                     "academic_style", "comprehensive_typing"]),
            ("general", ["perfect_formatting", "consistent_style", "generic_names", 
                        "complete_error_handling", "balanced_comments", "standard_structure", 
                        "predictable_patterns", "uniform_spacing"])
        ]:
            for pattern in patterns:
                pattern_names.append(f"pattern_{family}_{pattern}")
        names.extend(pattern_names)
        
        # Perplexity feature names (8) - if enabled
        if self.use_perplexity:
            ppl_names = [
                "ppl_qwen", "ppl_starcoder", "ppl_window_mean", "ppl_window_std",
                "ppl_window_range", "ppl_conditional", "ppl_cross_model_agreement",
                "ppl_token_variance"
            ]
            names.extend(ppl_names)
        
        # Stylometric feature names (12)
        style_names = [
            "style_avg_ident_len", "style_ident_entropy", "style_naming_consistency",
            "style_whitespace_entropy", "style_indent_consistency", "style_comment_density",
            "style_comment_variance", "style_line_len_variance", "style_token_diversity",
            "style_human_markers", "style_code_comment_ratio", "style_formatting_consistency"
        ]
        names.extend(style_names)
        
        return names
    
    def extract_with_names(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, float]:
        """
        Extract features and return as named dictionary.
        
        Useful for analysis and debugging.
        """
        features = self.extract(code, language)
        names = self.get_feature_names()
        
        return dict(zip(names, features))


if __name__ == "__main__":
    # Test unified extractor
    print("Testing Unified Feature Extractor")
    print("=" * 60)
    
    # Sample code
    test_code = """
def calculate_fibonacci(n: int) -> int:
    '''
    Calculate the nth Fibonacci number using iteration.
    
    This function computes Fibonacci numbers efficiently using
    an iterative approach with O(n) time complexity.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Example:
        >>> calculate_fibonacci(10)
        55
    '''
    if n <= 1:
        return n
    
    previous_value = 0
    current_value = 1
    
    for iteration_index in range(2, n + 1):
        next_value = previous_value + current_value
        previous_value = current_value
        current_value = next_value
    
    return current_value
"""
    
    print("\nInitializing extractor (without perplexity for speed)...")
    extractor = UnifiedFeatureExtractor(use_perplexity=False)
    
    print(f"\nTotal features: {extractor.get_feature_count()}")
    print(f"Expected: 33 (AST) + 57 (Pattern) + 12 (Style) = 102")
    
    print("\nExtracting features...")
    features = extractor.extract(test_code)
    
    print(f"\nExtracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Non-zero features: {np.count_nonzero(features)}")
    
    print("\nTop 10 strongest features:")
    feature_dict = extractor.extract_with_names(test_code)
    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_features[:10]:
        print(f"  {name:40s}: {value:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ Feature extraction test complete")
