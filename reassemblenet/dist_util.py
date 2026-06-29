"""
Helpers for distributed training.
"""

import io
import os
import platform
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

import warnings
warnings.filterwarnings('ignore')

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_single_process():
    return _world_size() == 1


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _backend():
    if platform.system() == "Windows":
        return "gloo"
    return "nccl" if th.cuda.is_available() else "gloo"


def setup_dist():
    """
    Setup a distributed process group.
    """
    #import pdb;pdb.set_trace()
    if dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    backend = _backend()

    if _is_single_process():
        return

    if MPI is None:
        dist.init_process_group(backend=backend, init_method="env://")
        return

    ## temporary removed to manually set the CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")
   

import torch
torch.cuda.device_count()
def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if _is_single_process() or not dist.is_initialized() or MPI is None:
        return th.load(path, **kwargs)

    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)
            #data += dist.broadcast(None)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def barrier():
    """
    Synchronize all ranks when running distributed.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return
    dist.barrier()


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
