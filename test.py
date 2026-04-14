import torch
import torchvision
import numpy as np
import matplotlib

def main():
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("NumPy version:", np.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("CUDA available:", torch.cuda.is_available())

if __name__ == "__main__":
    main()