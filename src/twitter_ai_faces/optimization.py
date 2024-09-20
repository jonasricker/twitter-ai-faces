import datetime
import functools
import os
import re

import joblib
import pandas as pd
from joblib import register_store_backend
from joblib._store_backends import CacheItemInfo, FileSystemStoreBackend
from joblib.memory import MemorizedFunc, NotMemorizedFunc


class PatchedMemorizedFunc(MemorizedFunc):
    def _check_previous_func_code(self, stacklevel=2):
        return True


class SimpleMemory(joblib.Memory):
    """
    Simplified version of joblib.Memory which ignores code changes.
    By default the cache is written to 'cache', can be controleld using the environment
    variable CACHEDIR. If CACHEDIR is set to an empty string, caching is disabled.
    """

    def __init__(
        self,
        location=None,
        backend="local",
        mmap_mode=None,
        compress=False,
        verbose=1,
        bytes_limit=None,
        backend_options=None,
    ):
        if location is None:
            location = os.environ.get("CACHEDIR", "cache")
        if location == "":
            location = None
        super().__init__(
            location,
            backend,
            mmap_mode,
            compress,
            verbose,
            bytes_limit,
            backend_options,
        )

    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False):
        """Decorates the given function func to only compute its return
        value for input arguments not cached on disk. Ignores function code.

        Parameters
        ----------
        func: callable, optional
            The function to be decorated
        ignore: list of strings
            A list of arguments name to ignore in the hashing
        verbose: integer, optional
            The verbosity mode of the function. By default that
            of the memory object is used.
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments. By default that of the memory object is used.

        Returns
        -------
        decorated_func: MemorizedFunc object
            The returned object is a MemorizedFunc object, that is
            callable (behaves like a function), but offers extra
            methods for cache lookup and management. See the
            documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(
                self.cache, ignore=ignore, verbose=verbose, mmap_mode=mmap_mode
            )
        if self.store_backend is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, MemorizedFunc):
            func = func.func
        return PatchedMemorizedFunc(
            func,
            location=self.store_backend,
            backend=self.backend,
            ignore=ignore,
            mmap_mode=mmap_mode,
            compress=self.compress,
            verbose=verbose,
            timestamp=self.timestamp,
        )


class PandasFileSystemStoreBackend(FileSystemStoreBackend):
    def load_item(self, path, verbose=1, msg=None):
        """Load an item from the store given its path as a list of
        strings."""
        full_path = os.path.join(self.location, *path)

        if verbose > 1:
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        filename = os.path.join(full_path, "output.feather")
        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        return pd.read_feather(filename)

    def dump_item(self, path, item, verbose=1):
        """Dump an item in the store at the path given as a list of
        strings."""
        item_path = os.path.join(self.location, *path)
        if not self._item_exists(item_path):
            self.create_location(item_path)
        filename = os.path.join(item_path, "output.feather")
        if verbose > 10:
            print("Persisting in %s" % item_path)

        def write_func(to_write, dest_filename):
            to_write.to_feather(dest_filename)

        self._concurrency_safe_write(item, filename, write_func)

    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of
        strings"""
        item_path = os.path.join(self.location, *path)
        filename = os.path.join(item_path, "output.feather")

        return self._item_exists(filename)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            is_cache_hash_dir = re.match("[a-f0-9]{32}", os.path.basename(dirpath))

            if is_cache_hash_dir:
                output_filename = os.path.join(dirpath, "output.feather")
                try:
                    last_access = os.path.getatime(output_filename)
                except OSError:
                    try:
                        last_access = os.path.getatime(dirpath)
                    except OSError:
                        # The directory has already been deleted
                        continue

                last_access = datetime.datetime.fromtimestamp(last_access)
                try:
                    full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                    dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
                except OSError:
                    # Either output_filename or one of the files in
                    # dirpath does not exist any more. We assume this
                    # directory is being cleaned by another process already
                    continue

                items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items


register_store_backend("pandas", PandasFileSystemStoreBackend)
