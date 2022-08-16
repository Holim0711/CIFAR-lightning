__all__ = ['read_labels']


try:  # python version >= 3.7
    from importlib.resources import read_text
except ImportError:
    from pkg_resources import resource_string

    def read_text(package, resource):
        return resource_string(package, resource).decode()


def read_labels(dataset: str):
    filename = dataset.upper() + '.txt'
    return list(filter(None, read_text(__package__, filename).split('\n')))
