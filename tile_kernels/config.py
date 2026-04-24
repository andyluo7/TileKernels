import functools
import torch

_num_sms = 0


@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.multi_processor_count


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms() -> int:
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms()
    return _num_sms


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    if hasattr(prop, 'shared_memory_per_multiprocessor'):
        return prop.shared_memory_per_multiprocessor
    # ROCm: attribute not available, use per-block shared memory as approximation
    # MI300X (gfx942): 64KB per CU, MI355X (gfx950): 64KB per CU
    return getattr(prop, 'max_shared_memory_per_block', 65536)
