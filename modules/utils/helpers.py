def str_to_bool(string: str) -> bool:
    if string == 'False':
        return False
    elif string == 'True':
        return True
    raise TypeError(f'String {string} cannot be casted to boolean')


def should_consider_image(section: int, meta: dict) -> bool:
    return (
        (section == 0 and str_to_bool(meta['LeftBack']))
        or (section == 1 and str_to_bool(meta['LeftLeft']))
        or (section == 2 and str_to_bool(meta['LeftFront']))
    )
