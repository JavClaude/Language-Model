from torch.cuda import is_available

CUDA = "cuda"
CPU = "cpu"

DEVICE = CUDA if is_available() else CPU
