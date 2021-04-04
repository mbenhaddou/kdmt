import collections
from decorator import decorator

def lazy_property_factory(lazy_property):
    """Create properties that perform lazy loading of attributes."""
    def lazy_property_getter(self):
        if not hasattr(self, '_' + lazy_property):
            self.load()
        if not hasattr(self, '_' + lazy_property):
            raise ValueError("{} wasn't loaded".format(lazy_property))
        return getattr(self, '_' + lazy_property)

    def lazy_property_setter(self, value):
        setattr(self, '_' + lazy_property, value)

    return lazy_property_getter, lazy_property_setter



def do_not_pickle_attributes(*lazy_properties):
    r"""Decorator to assign non-pickable properties.

    Used to assign properties which will not be pickled on some class.
    This decorator creates a series of properties whose values won't be
    serialized; instead, their values will be reloaded (e.g. from disk) by
    the :meth:`load` function after deserializing the object.

    The decorator can be used to avoid the serialization of bulky
    attributes. Another possible use is for attributes which cannot be
    pickled at all. In this case the user should construct the attribute
    himself in :meth:`load`.

    Parameters
    ----------
    \*lazy_properties : strings
        The names of the attributes that are lazy.

    Notes
    -----
    The pickling behavior of the dataset is only overridden if the
    dataset does not have a ``__getstate__`` method implemented.

    """
    def wrap_class(cls):
        if not hasattr(cls, 'load'):
            raise ValueError("no load method implemented")

        # Attach the lazy loading properties to the class
        for lazy_property in lazy_properties:
            setattr(cls, lazy_property,
                    property(*lazy_property_factory(lazy_property)))

        # Delete the values of lazy properties when serializing
        if not hasattr(cls, '__getstate__'):
            def __getstate__(self):
                serializable_state = self.__dict__.copy()
                for lazy_property in lazy_properties:
                    attr = serializable_state.get('_' + lazy_property)
                    # Iterators would lose their state
                    if isinstance(attr, collections.Iterator):
                        raise ValueError("Iterators can't be lazy loaded")
                    serializable_state.pop('_' + lazy_property, None)
                return serializable_state
            setattr(cls, '__getstate__', __getstate__)

        return cls
    return wrap_class


def requires_properties(properties):
    @decorator
    def _requires_properties(func, *args, **kwargs):
        params = util.map_parameters_in_fn_call(args, kwargs, func)
        obj = params.get('self')

        if obj is None:
            raise Exception('This decorator only works on instance methods')

        missing = [p for p in properties if getattr(obj, p) is None]

        if len(missing):
            raise ValueError('{} requires {} to be set, missing: {}'
                             .format(func.__name__, properties, missing))

        return func(*args, **kwargs)

    return _requires_properties
