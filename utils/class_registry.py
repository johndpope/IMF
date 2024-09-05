import inspect
import typing
import omegaconf
import dataclasses


class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def make_dataclass_from_init(self, func, name, arg_keys, stop_args):
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                dataclasses.field(default=v.default),
            )
            for k, v in args.items()
        ]
        args = [arg for arg in args if arg[0] not in stop_args]
        if arg_keys:
            self.arg_keys = arg_keys
            arg_classes = dict()
            for key in arg_keys:
                arg_classes[key] = dataclasses.make_dataclass(key, args)
            return dataclasses.make_dataclass(
                name,
                [
                    (k, v, dataclasses.field(default=v()))
                    for k, v in arg_classes.items()
                ],
            )
        return dataclasses.make_dataclass(name, args)

    def make_dataclass_from_classes(self, name):
        return dataclasses.make_dataclass(
            name,
            [(k, v, dataclasses.field(default=v())) for k, v in self.classes.items()],
        )

    def make_dataclass_from_args(self, name):
        return dataclasses.make_dataclass(
            name,
            [(k, v, dataclasses.field(default=v())) for k, v in self.args.items()],
        )

    def add_to_registry(
        self, name, arg_keys=None, stop_args=("self", "args", "kwargs")
    ):
        def add_class_by_name(cls):
            self.classes[name] = cls
            self.args[name] = self.make_dataclass_from_init(
                cls.__init__, name, arg_keys, stop_args
            )
            return cls

        return add_class_by_name
