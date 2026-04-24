def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def get_warp_size() -> int:
    """Get the hardware warp/wavefront size for the current target.
    
    NVIDIA: 32 threads per warp
    AMD ROCm: 64 threads per wavefront (wave64)
    """
    try:
        from tilelang.utils.target import determine_target
        target = determine_target(return_object=True)
        target_kind = target.kind.name if hasattr(target, 'kind') else ''
        if target_kind == 'hip':
            return 64  # AMD wave64
        # Check attrs for thread_warp_size
        attrs = getattr(target, 'attrs', {})
        if hasattr(attrs, 'get'):
            return attrs.get('thread_warp_size', 32)
        return 32  # Default NVIDIA
    except Exception:
        return 32


def is_hip_target() -> bool:
    """Check if the current TileLang target is HIP/ROCm."""
    try:
        from tilelang.utils.target import determine_target
        target = determine_target(return_object=True)
        return target.kind.name == 'hip'
    except Exception:
        return False
