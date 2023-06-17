from collections import OrderedDict
from typing import Any, Hashable


class LRUCache:
    def __init__(self, maxsize: int = 100) -> None:
        """Initialize an empty LRU cache with a maximum size.

        Args:
            max_size (int, optional): Maximum size of the cache. Defaults to 100.
        """
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self._size = 0

    def __len__(self) -> int:
        """Return the size of the cache.

        Returns:
            int: Size of the cache.
        """
        return self._size

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is in the cache.

        Args:
            key (Hashable): Key to check.

        Returns:
            bool: True if the key is in the cache, False otherwise.
        """
        return key in self._cache

    def __getitem__(self, key: Hashable) -> object:
        """Get the value associated with a key.

        Args:
            key (Hashable): Key to get the value for.

        Returns:
            object: Value associated with the key.
        """
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Set the value associated with a key.

        Args:
            key (Hashable): Key to set the value for.
            value (Any): Value to set.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            self._cache[key] = value
            self._size += 1
            if self._size > self.maxsize:
                self._cache.popitem(last=False)
                self._size -= 1

    def __delitem__(self, key: Hashable) -> None:
        """Delete a key from the cache.

        Args:
            key (Hashable): Key to delete.
        """
        del self._cache[key]
        self._size -= 1

    def __iter__(self) -> Any:
        """Iterate over the keys in the cache.

        Returns:
            Any: Iterator over the keys in the cache.
        """
        return iter(self._cache)

    def __repr__(self) -> str:
        """Return a string representation of the cache.

        Returns:
            str: String representation of the cache.
        """
        return repr(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._size = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Get the value associated with a key.

        Args:
            key (Hashable): Key to get the value for.
            default (Any, optional): Default value to return if the key is not in the cache. Defaults to None.

        Returns:
            Any: Value associated with the key.
        """
        self._cache.move_to_end(key)
        return self._cache.get(key, default)

    def pop(self, key: Hashable, default: Any = None) -> Any:
        """Pop the value associated with a key.

        Args:
            key (Hashable): Key to pop the value for.
            default (Any, optional): Default value to return if the key is not in the cache. Defaults to None.

        Returns:
            Any: Value associated with the key.
        """
        value = self._cache.pop(key, default)
        if value is not default:
            self._size -= 1
        return value

    def items(self) -> Any:
        """Get the items in the cache.

        Returns:
            Any: Items in the cache.
        """
        return self._cache.items()
