from typing import Dict, Any, Generic, TypeVar, Type, Callable, Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


class Parser(ABC):
    @abstractmethod
    def parse(self, args_str: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def help(self) -> str:
        pass


@dataclass(frozen=True)
class ArgumentSpec:
    arg_name_parse: str
    arg_name: str
    value_type: Optional[Callable[[str], Any]]
    help_msg: str
    default: Optional[Any]

    @property
    def is_boolean(self) -> bool:
        return self.value_type is None


class DelimParser(Parser):
    def __init__(self) -> None:
        self._args: Dict[str, ArgumentSpec] = dict()

    @property
    @abstractmethod
    def delim(self) -> str:
        pass

    def add_argument(self, arg_name_parse: str, arg_name: str, value_type: Optional[Callable[[str], Any]], help_msg: str,
                     default: Optional[Any] = None) -> None:
        """

        :param arg_name_parse: Name on command line
        :param arg_name: Name of argument given to class
        :param value_type: Type
        :param help_msg: Help message
        :param default: Default value.
        :return: None
        """
        arg_name_parse = arg_name_parse.lower()
        assert (len(arg_name_parse) > 0) and (len(arg_name) > 0), "length of argument names must be > 0"
        assert self.delim not in arg_name_parse, f"Cannot have delimiter {self.delim} in {arg_name_parse}"

        assert arg_name_parse not in self._args.keys(), f"{arg_name_parse} already exists"

        self._args[arg_name_parse] = ArgumentSpec(arg_name_parse, arg_name, value_type, help_msg, default)

    def parse(self, args_str: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict()
        if len(args_str) == 0:
            return kwargs

        args_str_l: List[str] = args_str.split(self.delim)
        for arg_str_i in args_str_l:
            assert len(arg_str_i) > 0, f"Empty argument in {args_str}. Perhaps problem with delimiters."

            arg_name_parse, arg_value_str = self._match_arg_name(arg_str_i)
            arg_spec: ArgumentSpec = self._args[arg_name_parse]
            if arg_spec.is_boolean:
                assert len(arg_value_str) == 0, (f"boolean arguments should not have preceding value. {arg_str_i} has arg name {arg_name_parse} and value "
                                                 f"{arg_value_str}")
                kwargs[arg_spec.arg_name] = True
            else:
                assert len(arg_value_str) > 0, (f"non-boolean argument {arg_str_i} does not have preceding value. Parsed '{arg_name_parse}' for name and "
                                                f"'{arg_value_str}' for value")
                assert arg_spec.value_type is not None
                kwargs[arg_spec.arg_name] = arg_spec.value_type(arg_value_str)

        for arg_spec in self._args.values():
            if (arg_spec.arg_name not in kwargs.keys()) and (arg_spec.default is not None):
                kwargs[arg_spec.arg_name] = arg_spec.default

        return kwargs

    def help(self) -> str:
        help_str_l: List[str] = ["Format <arg_val><arg_name> for non-boolean arguments and <arg_name> for boolean arguments. <arg_name> is case invariant.",
                                 f"Delimited by {self.delim}."]
        for arg_spec in self._args.values():
            if arg_spec.is_boolean:
                usage = arg_spec.arg_name_parse
            else:
                type_name = getattr(arg_spec.value_type, "__name__", str(arg_spec.value_type))
                usage = f"<{type_name}>{arg_spec.arg_name_parse}"
            help_i = f"{usage}: {arg_spec.help_msg}"

            if arg_spec.default is not None:
                help_i = f"{help_i} (Default: {arg_spec.default})"

            help_str_l.append(help_i)

        return "\n".join(help_str_l)

    def _match_arg_name(self, arg_str_i: str) -> Tuple[str, str]:
        """
        Match to the longest argument name (case-insensitive).
        """

        token_lower = arg_str_i.lower()

        matches = [name for name in self._args.keys() if token_lower.endswith(name)]

        if len(matches) == 0:
            raise ValueError(f"Unexpected argument {arg_str_i!r}")

        arg_name: str = max(matches, key=len)
        arg_value = arg_str_i[: -len(arg_name)]

        return arg_name, arg_value


T = TypeVar("T")


class Factory(Generic[T]):
    def __init__(self, class_type_str: str):
        self._class_registry: Dict[str, Type[T]] = dict()
        self._parser_registry: Dict[str, Type[Parser]] = dict()
        self._class_type_str: str = class_type_str

    def register_class(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._class_registry.keys():
                raise ValueError(f"{self._class_type_str.capitalize()} {name!r} already registered")
            self._class_registry[name] = cls
            return cls
        return deco

    def register_parser(self, name: str) -> Callable[[Type[Parser]], Type[Parser]]:
        def deco(cls: Type[Parser]) -> Type[Parser]:
            if name in self._parser_registry.keys():
                raise ValueError(f"{self._class_type_str.capitalize()} parser {name!r} already registered")
            self._parser_registry[name] = cls
            return cls
        return deco

    def get_parser(self, name: str) -> Optional[Parser]:
        cls_parser: Optional[Type[Parser]] = self._parser_registry.get(name)
        if cls_parser is not None:
            return cls_parser()
        else:
            return None

    def get_kwargs(self, name: str, args_str: Optional[str]) -> Dict[str, Any]:
        self.get_type(name)
        kwargs: Dict[str, Any] = dict()
        parser: Optional[Parser] = self.get_parser(name)
        if (parser is not None) and (args_str is not None):
            try:
                kwargs = parser.parse(args_str)
            except Exception as e:
                logging.exception(f"Error occurred: {e}")
                raise ValueError(f"Error parsing {args_str} for {self._class_type_str} {name!r}.\nParser help:\n{parser.help()}")
        else:
            assert args_str is None, f"No parser for {self._class_type_str} {name}, however, args given are {args_str}"
        return kwargs

    def get_type(self, name: str) -> Type[T]:
        try:
            return self._class_registry[name]
        except KeyError:
            raise ValueError(f"Unknown {self._class_type_str} {name!r}. Available: {sorted(self._class_registry)}")

    def build_class(self, name: str, kwargs: Dict[str, Any]) -> T:
        cls: Type[T] = self.get_type(name)
        return cls(**kwargs)

    def get_all_class_names(self) -> List[str]:
        return list(self._class_registry.keys())
