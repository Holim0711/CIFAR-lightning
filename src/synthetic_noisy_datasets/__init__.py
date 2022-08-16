try:  # python version >= 3.7
    from importlib.resources import read_text
except ImportError:
    from pkg_resources import resource_string

    def read_text(package, resource):
        return resource_string(package, resource).decode()


