"""Tool registry: plain Python functions become model-callable tools.

The platform image does not ship the `mcp` package, so the loop offers tools to
the model directly in the OpenAI ``tools=[...]`` format and dispatches the calls
itself. This module is that layer: a decorator turns an annotated function into
a tool, its signature becomes the JSON schema, and its docstring becomes the
description the model reads.

The contract is kept MCP-shaped on purpose. ``name`` / ``description`` /
``parameters`` is exactly what an MCP server advertises, so the same registry can
be exposed over a real MCP server for local debugging without touching a tool.

Two properties matter more than features here:

* **One source of truth.** The signature *is* the schema. A tool cannot drift
  from its advertised contract, because there is nothing to keep in sync.
* **Failure isolation.** A tool that raises returns a structured, readable error
  instead of killing the run. Experimental mods additionally disable themselves,
  so a half-finished idea cannot cost the team a submission.
"""

from __future__ import annotations

import inspect
import json
import re
import traceback
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin

#: Operating regimes. The agent declares one early; tools can opt into either.
#: ``standard`` is the declarative fast path, ``custom`` the free-code escape.
Regime = Literal["standard", "custom"]
ALL_REGIMES: frozenset[str] = frozenset({"standard", "custom"})

_PRIMITIVES: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


class ToolError(Exception):
    """A refusal the model is expected to act on.

    Carries a hint, and optionally a corrected payload. Handing back a ready-to-send
    correction instead of only a description of the problem is what turns a dead
    end into one more call.
    """

    def __init__(self, problem: str, hint: str = "", fix: Any = None, code: str = "tool_error"):
        super().__init__(problem)
        self.problem, self.hint, self.fix, self.code = problem, hint, fix, code

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": "error", "code": self.code, "problem": self.problem}
        if self.hint:
            out["hint"] = self.hint
        if self.fix is not None:
            out["corrected_input"] = self.fix
            out["next_step"] = "Call this tool again with corrected_input exactly as given."
        return out


# ---------------------------------------------------------------------------
# Docstring -> descriptions
# ---------------------------------------------------------------------------

_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_ARG_LINE = re.compile(r"^\s{2,}(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a summary and per-argument descriptions.

    Keeping argument docs in the docstring rather than in decorator metadata means
    the text the model reads sits next to the code it describes.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    args: dict[str, str] = {}
    current: str | None = None
    in_args = False

    for line in lines:
        if _ARGS_HEADER.match(line):
            in_args = True
            continue
        if in_args:
            if line.strip() and not line.startswith((" ", "\t")):
                in_args = False  # a new unindented section ends the Args block
            else:
                match = _ARG_LINE.match(line)
                if match:
                    current = match.group(1).lstrip("*")
                    args[current] = match.group(2).strip()
                    continue
                if current and line.strip():
                    args[current] += " " + line.strip()
                    continue
        if not in_args:
            summary.append(line)

    return "\n".join(summary).strip(), args


# ---------------------------------------------------------------------------
# Signature -> JSON schema
# ---------------------------------------------------------------------------


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _PRIMITIVES:
        return dict(_PRIMITIVES[annotation])

    origin, args = get_origin(annotation), get_args(annotation)

    if origin is Literal:
        kinds = {type(a) for a in args}
        base = _PRIMITIVES.get(kinds.pop()) if len(kinds) == 1 else None
        return {**(base or {"type": "string"}), "enum": list(args)}

    if origin in (Union, getattr(typing, "UnionType", Union)):
        inner = [a for a in args if a is not type(None)]
        return _annotation_schema(inner[0]) if len(inner) == 1 else {}

    if origin in (list, set, tuple):
        item = _annotation_schema(args[0]) if args else {}
        return {"type": "array", **({"items": item} if item else {})}

    if origin is dict or annotation is dict:
        return {"type": "object", "additionalProperties": True}

    return {}


def schema_from_signature(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    """Derive ``(description, json_schema)`` from a function's signature and docstring."""
    summary, arg_docs = parse_docstring(fn.__doc__)
    try:
        hints = typing.get_type_hints(fn)
    except Exception as err:
        # Fail loudly rather than degrade to an untyped schema. A schema without
        # types still "works": the model simply loses the enums and the argument
        # types, silently, and only its behaviour reveals it. In practice this
        # means an annotation was imported inside a function instead of at module
        # level, which is worth one clear error at startup.
        raise ValueError(
            f"tool '{fn.__name__}': its type annotations could not be resolved ({err}). "
            f"Import the annotation types at module level so the schema can be derived."
        ) from None

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        prop = _annotation_schema(hints.get(name, param.annotation))
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            if isinstance(param.default, (str, int, float, bool)) or param.default is None:
                prop["default"] = param.default
        properties[name] = prop

    schema = {"type": "object", "properties": properties, "required": required}
    return summary, schema


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    regimes: frozenset[str] = ALL_REGIMES
    experimental: bool = False
    enabled: bool = True
    #: Set when an experimental mod fails; surfaced in diagnostics.
    disabled_reason: str = ""

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    # -- registration ------------------------------------------------------

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        regimes: typing.Iterable[str] = ALL_REGIMES,
        experimental: bool = False,
        enabled: bool = True,
    ) -> Any:
        """Register a function as a tool. Usable bare or with arguments."""

        def wrap(target: Callable[..., Any]) -> Callable[..., Any]:
            description, parameters = schema_from_signature(target)
            tool_name = name or target.__name__
            if not description:
                raise ValueError(
                    f"tool '{tool_name}' has no docstring; the model reads it as the "
                    f"tool description, so it is required"
                )
            self.tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                parameters=parameters,
                fn=target,
                regimes=frozenset(regimes),
                experimental=experimental,
                enabled=enabled,
            )
            return target

        return wrap(fn) if callable(fn) else wrap

    # -- introspection -----------------------------------------------------

    def schemas(self, regime: str | None = None) -> list[dict[str, Any]]:
        """OpenAI ``tools=[...]`` payload, filtered to what applies right now."""
        return [
            t.as_openai_tool()
            for t in self.tools.values()
            if t.enabled and (regime is None or regime in t.regimes)
        ]

    def names(self, regime: str | None = None) -> list[str]:
        return [
            t.name
            for t in self.tools.values()
            if t.enabled and (regime is None or regime in t.regimes)
        ]

    def disabled(self) -> dict[str, str]:
        return {t.name: t.disabled_reason for t in self.tools.values() if not t.enabled}

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, name: str, arguments: dict[str, Any] | str | None) -> str:
        """Execute a tool call and return the JSON string the model will read.

        Never raises: every failure becomes a structured, actionable message. A
        crash here would end the run, and an unfinished run scores zero.
        """
        tool = self.tools.get(name)
        if tool is None or not tool.enabled:
            return _dump(self._unknown_tool(name, tool))

        try:
            kwargs = _coerce_arguments(arguments)
        except ToolError as err:
            return _dump(err.payload())

        try:
            _check_arguments(tool, kwargs)
            result = tool.fn(**kwargs)
        except ToolError as err:
            return _dump(err.payload())
        except Exception as err:  # noqa: BLE001 - the loop must survive anything
            if tool.experimental:
                tool.enabled = False
                tool.disabled_reason = f"{type(err).__name__}: {err}"
            return _dump(
                {
                    "status": "error",
                    "code": "exception",
                    "problem": f"{type(err).__name__}: {err}",
                    "traceback": traceback.format_exc(limit=3)[-1500:],
                    **(
                        {"note": f"Experimental tool '{name}' has been disabled for this run."}
                        if tool.experimental
                        else {}
                    ),
                }
            )

        return result if isinstance(result, str) else _dump(result)

    def _unknown_tool(self, name: str, tool: Tool | None) -> dict[str, Any]:
        if tool is not None:
            return {
                "status": "error",
                "code": "tool_disabled",
                "problem": f"tool '{name}' is disabled: {tool.disabled_reason or 'turned off'}",
                "hint": f"Available: {self.names()}",
            }
        import difflib

        near = difflib.get_close_matches(name, list(self.tools), n=1, cutoff=0.5)
        return {
            "status": "error",
            "code": "unknown_tool",
            "problem": f"there is no tool named '{name}'",
            "hint": (f"Did you mean '{near[0]}'? " if near else "") + f"Available: {self.names()}",
        }


def _coerce_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ToolError(
                f"arguments were not valid JSON: {err}",
                "Re-send the call with a valid JSON object.",
                code="bad_json",
            ) from None
    if not isinstance(arguments, dict):
        raise ToolError(
            f"arguments must be a JSON object, got {type(arguments).__name__}",
            'Example: {"dataset": "train"}',
            code="bad_arguments",
        )
    return arguments


def _check_arguments(tool: Tool, kwargs: dict[str, Any]) -> None:
    properties = tool.parameters.get("properties", {})
    missing = [r for r in tool.parameters.get("required", []) if r not in kwargs]
    unknown = [k for k in kwargs if k not in properties]
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing required: {missing}")
        if unknown:
            problems.append(f"unexpected: {unknown}")
        raise ToolError(
            f"bad arguments for '{tool.name}' -- " + "; ".join(problems),
            f"Accepted parameters: {sorted(properties)}.",
            code="bad_arguments",
        )


def _dump(value: Any) -> str:
    """Compact JSON. Indentation would cost roughly a third of the tokens in whitespace."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


#: The process-wide registry. Core tools and mods both register against it.
REGISTRY = Registry()
tool = REGISTRY.tool
