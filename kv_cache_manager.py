import torch
import os
import json
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

__author__ = "eddy"

class KVCacheManager:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            cache_dir = os.path.join(current_dir, ".kv_cache")

        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise

        self.metadata_file = self.cache_dir / "compilation_metadata.json"
        self.metadata = self._load_metadata()

        logging.info(f"Compilation tracker initialized at: {self.cache_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save cache metadata: {e}")

    def _compute_model_hash(self, model_path: str, config: Dict[str, Any]) -> str:
        hasher = hashlib.sha256()

        hasher.update(model_path.encode('utf-8'))

        if os.path.exists(model_path):
            stat = os.stat(model_path)
            hasher.update(str(stat.st_size).encode('utf-8'))
            hasher.update(str(stat.st_mtime).encode('utf-8'))

        config_str = json.dumps(config, sort_keys=True)
        hasher.update(config_str.encode('utf-8'))

        hasher.update(torch.__version__.encode('utf-8'))
        if torch.cuda.is_available():
            hasher.update(torch.version.cuda.encode('utf-8'))

        return hasher.hexdigest()

    def check_compiled_cache(self, model_path: str, config: Dict[str, Any]) -> bool:
        cache_key = self._compute_model_hash(model_path, config)

        if cache_key in self.metadata:
            cache_info = self.metadata[cache_key]
            logging.info(f"Compilation cache exists for model: {os.path.basename(model_path)}")
            logging.info(f"Previously compiled on: {cache_info.get('timestamp', 'unknown')}")
            return True

        logging.info(f"No compilation cache found for model: {os.path.basename(model_path)}")
        return False

    def mark_compiled(self, model_path: str, config: Dict[str, Any]):
        cache_key = self._compute_model_hash(model_path, config)

        try:
            from datetime import datetime
            self.metadata[cache_key] = {
                "model_path": model_path,
                "config": config,
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            }
            self._save_metadata()

            logging.info(f"Marked model as compiled: {os.path.basename(model_path)}")
        except Exception as e:
            logging.error(f"Failed to mark model as compiled: {e}")

    def clear_cache(self, older_than_days: Optional[int] = None):
        from datetime import datetime, timedelta

        cleared_count = 0

        for cache_key, cache_info in list(self.metadata.items()):
            should_clear = False

            if older_than_days is not None:
                try:
                    timestamp = datetime.fromisoformat(cache_info.get('timestamp', ''))
                    age = datetime.now() - timestamp
                    if age > timedelta(days=older_than_days):
                        should_clear = True
                except:
                    pass
            else:
                should_clear = True

            if should_clear:
                del self.metadata[cache_key]
                cleared_count += 1

        self._save_metadata()
        logging.info(f"Cleared {cleared_count} compilation metadata entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "total_compiled_models": len(self.metadata),
            "cache_directory": str(self.cache_dir),
        }


class TorchCompileCache:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            cache_dir = os.path.join(current_dir, ".torch_compile_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self._setup_torch_compile_cache()

        logging.info(f"Torch Compile Cache initialized at: {self.cache_dir}")

    def _setup_torch_compile_cache(self):
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 256

        if hasattr(torch._dynamo.config, 'cache_dir'):
            torch._dynamo.config.cache_dir = str(self.cache_dir)

        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(self.cache_dir)

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)

        logging.info("Torch compile cache configured for persistent caching")

    def clear_cache(self):
        import shutil
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                logging.info("Torch compile cache cleared")
        except Exception as e:
            logging.error(f"Failed to clear torch compile cache: {e}")

    def get_cache_size(self) -> float:
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)


class AttentionKVCache:
    def __init__(self, max_batch_size: int = 1, max_seq_length: int = 4096):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.cache = {}
        self.enabled = False

        logging.info(f"Attention KV Cache initialized: batch_size={max_batch_size}, seq_len={max_seq_length}")

    def enable(self):
        self.enabled = True
        logging.info("Attention KV Cache enabled")

    def disable(self):
        self.enabled = False
        logging.info("Attention KV Cache disabled")

    def clear(self):
        self.cache.clear()
        logging.info("Attention KV Cache cleared")

    def allocate_cache(self, layer_idx: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        if not self.enabled:
            return None

        if layer_idx not in self.cache:
            k_cache = torch.zeros(
                (self.max_batch_size, num_heads, self.max_seq_length, head_dim),
                device=device,
                dtype=dtype
            )
            v_cache = torch.zeros(
                (self.max_batch_size, num_heads, self.max_seq_length, head_dim),
                device=device,
                dtype=dtype
            )
            self.cache[layer_idx] = {"k": k_cache, "v": v_cache, "seq_len": 0}
            logging.debug(f"Allocated KV cache for layer {layer_idx}")

        return self.cache[layer_idx]

    def get_cache(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        return self.cache.get(layer_idx)

    def update_cache(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, seq_len: int):
        if layer_idx in self.cache:
            cache = self.cache[layer_idx]
            cache["k"][:, :, :seq_len, :] = k
            cache["v"][:, :, :seq_len, :] = v
            cache["seq_len"] = seq_len

    def get_memory_usage_mb(self) -> float:
        total_bytes = 0
        for cache_dict in self.cache.values():
            total_bytes += cache_dict["k"].numel() * cache_dict["k"].element_size()
            total_bytes += cache_dict["v"].numel() * cache_dict["v"].element_size()
        return total_bytes / (1024 * 1024)


class UnifiedCacheManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        current_dir = os.path.dirname(os.path.realpath(__file__))
        base_cache_dir = os.path.join(current_dir, ".unified_cache")

        self.kv_cache_manager = KVCacheManager(os.path.join(base_cache_dir, "kv_cache"))
        self.compile_cache = TorchCompileCache(os.path.join(base_cache_dir, "compile_cache"))
        self.attention_cache = AttentionKVCache(max_batch_size=1, max_seq_length=4096)

        self._initialized = True
        logging.info("Unified Cache Manager initialized")

    def check_and_mark_compilation(self, model_path: str, compile_config: Dict[str, Any]) -> bool:
        is_cached = self.kv_cache_manager.check_compiled_cache(model_path, compile_config)
        return is_cached

    def mark_model_compiled(self, model_path: str, compile_config: Dict[str, Any]):
        self.kv_cache_manager.mark_compiled(model_path, compile_config)

    def clear_all_caches(self):
        logging.info("Clearing all caches...")
        self.kv_cache_manager.clear_cache()
        self.compile_cache.clear_cache()
        self.attention_cache.clear()
        logging.info("All caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        stats = {
            "kv_cache": self.kv_cache_manager.get_cache_stats(),
            "compile_cache_size_mb": self.compile_cache.get_cache_size(),
            "attention_cache_memory_mb": self.attention_cache.get_memory_usage_mb(),
        }
        return stats

    def print_cache_stats(self):
        stats = self.get_cache_stats()

        print("=" * 80)
        print("Cache Statistics")
        print("=" * 80)
        print()

        print("Compilation Tracker:")
        print(f"  Tracked compilations: {stats['kv_cache']['total_compiled_models']}")
        print(f"  Metadata directory: {stats['kv_cache']['cache_directory']}")
        print()

        print("Torch Inductor Cache:")
        print(f"  Size: {stats['compile_cache_size_mb']:.2f} MB")
        print()

        print("Attention KV Cache:")
        print(f"  Memory usage: {stats['attention_cache_memory_mb']:.2f} MB")
        print()

        print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cache_manager = UnifiedCacheManager()
    cache_manager.print_cache_stats()
