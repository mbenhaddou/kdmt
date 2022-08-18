from typing import Dict, TypeVar
T = TypeVar('T')
U = TypeVar('U')

class NonMutableDict(Dict[T, U]):
  """Dict where keys can only be added but not modified.

  Raises an error if a key is overwritten. The error message can be customized
  during construction. It will be formatted using {key} for the overwritten key.
  """

  def __init__(self, *args, **kwargs):
    self._error_msg = kwargs.pop(
        'error_msg',
        'Try to overwrite existing key: {key}',
    )
    if kwargs:
      raise ValueError('NonMutableDict cannot be initialized with kwargs.')
    super(NonMutableDict, self).__init__(*args, **kwargs)

  def __setitem__(self, key, value):
    if key in self.keys():
      raise ValueError(self._error_msg.format(key=key))
    return super(NonMutableDict, self).__setitem__(key, value)

  def update(self, other):
    if any(k in self.keys() for k in other):
      raise ValueError(self._error_msg.format(key=set(self) & set(other)))
    return super(NonMutableDict, self).update(other)