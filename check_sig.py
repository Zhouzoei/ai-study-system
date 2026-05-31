from engines.pipeline import EnhancedRAGPipeline
import inspect
sig = inspect.signature(EnhancedRAGPipeline.ask)
print(sig)
