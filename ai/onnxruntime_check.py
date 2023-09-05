import mxnet as mx
print(mx.context.num_gpus())

import faiss
print(faiss.has_gpu_support)
