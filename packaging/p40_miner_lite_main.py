"""Frozen entry point for the torch-free miner. cuda_capi handles adding the
bundled DLL directory (p40cuda.dll + cudart) to the search path."""
from miner_capi import main

if __name__ == "__main__":
    main()
