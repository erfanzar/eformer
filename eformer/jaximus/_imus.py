# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import abc
import dataclasses
import functools as ft
import itertools as it
import os
import typing as tp
import warnings
from abc import ABC
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, is_dataclass
from typing import TypeGuard

import chex
import jax
import jax._src
import jax._src.lax
import jax._src.pjit
import jax.core
import jax.core as core
import jax.extend
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import plum
from jax.custom_derivatives import SymbolicZero as SZ

WARN_ON_MATTER = os.environ.get("WARN_ON_MATTER", "true") in ["true", "yes", "1", "on"]
CT = tp.TypeVar("CT", bound=Callable)

StaticScalar = np.bool_ | np.number | bool | int | float | complex
ArrayLike = jax.Array | np.ndarray | StaticScalar


class OrginArray(ABC): ...  # noqa


OrginArray.register(jax.Array)


def ste(func):
    def _call(x: jax.Array | OrginArray, *args, **kwargs):
        zero = x - jax.lax.stop_gradient(x)
        y = jax.lax.stop_gradient(func(x, *args, **kwargs))
        return zero + y

    return _call


def default_handler(primitive, *args, **params):
    subfuns, bind_params = primitive.get_bind_params(params)
    return primitive.bind(*subfuns, *args, **bind_params)


def _materialize_all(vals):
    outs = []
    for val in vals:
        if hasattr(val, "materialize"):
            val = val.materialize()
        outs.append(val)
    return outs


def materialize_handler(primitive, *vals, params):
    vals = _materialize_all(vals)
    subfuns, bind_params = primitive.get_bind_params(params)
    result = primitive.bind(*subfuns, *vals, **bind_params)
    return result


class UninitializedAval(Exception): ...


def aux_field(metadata=None, **kwargs):
    metadata = dict(metadata) if metadata else {}
    metadata["implicit_array_aux"] = True
    return field(metadata=metadata, **kwargs)


class _AvalDescriptor:
    def __set_name__(self, owner, name):
        self._name = f"_{name}"

    def __get__(self, obj, owner=None):
        if obj is None:
            return None
        result = getattr(obj, self._name, None)
        if result is None:
            raise UninitializedAval()
        return result

    def __set__(self, obj, value):
        setattr(obj, self._name, value)


_aval_discovery = ContextVar("aval_discovery", default=False)


def _def_leaf(x):
    return isinstance(x, ImplicitArray)


def use_implicit(fn):
    def implicit_f(*args, **kwargs):
        leaves, struct = tu.tree_flatten(
            (fn, args, kwargs),
            is_leaf=_def_leaf,
        )

        tag = core.TraceTag()
        with core.take_current_trace() as ctrace:
            trace = _CustomTrace(parent_trace=ctrace, tag=tag)
            leaves = tu.tree_map(
                ft.partial(_wrap_tracer, trace=trace),
                leaves,
                is_leaf=_def_leaf,
            )
            func, args, kwargs = tu.tree_unflatten(struct, leaves)
            with core.set_current_trace(trace):
                outs = func(*args, **kwargs)
            outs = tu.tree_map(ft.partial(_unwrap_tracer, trace=trace), outs)
        return outs

    return implicit_f


implicit = use_implicit


def materialize_nested(implicit_arr, full=False):
    while isinstance(implicit_arr, ImplicitArray):
        try:
            implicit_arr = implicit_arr.materialize()
        except Exception:
            aval = implicit_arr.aval
            implicit_arr = jnp.ones(aval.shape, aval.dtype)
            break
        if not full:
            break
    return implicit_arr


def _get_materialization_aval(imp_arr):
    with _aval_discovery_context():
        result = jax.eval_shape(ft.partial(materialize_nested, full=True), imp_arr)
    return result


@contextmanager
def _aval_discovery_context():
    token = _aval_discovery.set(True)
    try:
        yield
    finally:
        _aval_discovery.reset(token)


def is_array(element: tp.Any) -> bool:
    return isinstance(element, np.ndarray | np.generic | jax.Array)


@dataclass
class _ArrayBase(OrginArray, abc.ABC):
    commute_ops: tp.ClassVar[bool] = True
    warn_on_materialize: tp.ClassVar[bool] = True

    default_shape: tp.ClassVar[tp.Sequence[int] | None] = None
    default_dtype: tp.ClassVar[jnp.dtype | None] = None

    shape: tp.Sequence[int] | None = aux_field(kw_only=True, default=None)
    dtype: jnp.dtype = aux_field(kw_only=True, default=None)  # noqa


@dataclass
class ImplicitArray(_ArrayBase):
    shape = _AvalDescriptor()
    dtype = _AvalDescriptor()

    def __post_init__(self):
        try:
            aval = _get_materialization_aval(self)
        except UninitializedAval:
            aval = None
        shape = None
        try:
            shape = self.shape
        except UninitializedAval:
            shape = self.shape = self.compute_shape()

        if aval is not None:
            if shape is None:
                self.shape = aval.shape
            elif shape != aval.shape:
                warnings.warn(
                    f"ImplicitArray shape {shape} does not match materialization shape {aval.shape}",
                    stacklevel=1,
                )
        elif shape is None:
            raise UninitializedAval("shape")

        dtype = None
        try:
            dtype = self.dtype
        except UninitializedAval:
            dtype = self.dtype = self.compute_dtype()

        if dtype is None and aval is None:
            aval = _get_materialization_aval(self)

        if aval is not None:
            if dtype is None:
                self.dtype = aval.shape
            elif dtype != aval.dtype:
                warnings.warn(
                    f"ImplicitArray dtype {dtype} does not match materialization dtype {aval.dtype}",
                    stacklevel=1,
                )
        elif dtype is None:
            raise UninitializedAval("dtype")

    def compute_shape(self):
        return self.default_shape

    def compute_dtype(self):
        return self.default_dtype

    @property
    def aval(self):
        return core.ShapedArray(self.shape, self.dtype)

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        if params is None:
            params = {}
        return materialize_handler(primitive, *args, params=params)

    @abc.abstractmethod
    def materialize(self): ...

    def tree_flatten_with_keys(self):
        children = []
        aux_data = []
        for name, is_aux in _get_names_and_aux(self):
            try:
                value = getattr(self, name)
            except UninitializedAval:
                if not _aval_discovery.get():
                    raise
                value = None
            if is_aux:
                aux_data.append(value)
            else:
                children.append((tu.GetAttrKey(name), value))

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        child_it = iter(children)
        aux_it = iter(aux_data)
        obj = cls.__new__(cls)
        for name, is_aux in _get_names_and_aux(cls):
            value = next(aux_it if is_aux else child_it)
            setattr(obj, name, value)

        return obj

    def astype(self, new_dtype):
        self.dtype = new_dtype
        return self

    def __init_subclass__(cls, commute_ops=True, warn_on_materialize=True, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.commute_ops = commute_ops
        cls.warn_on_materialize = warn_on_materialize

        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        core.pytype_aval_mappings[cls] = lambda x: x.aval
        tu.register_pytree_with_keys_class(cls)
        return cls


def _get_names_and_aux(obj):
    for val in dataclasses.fields(obj):
        yield val.name, bool(val.metadata.get("implicit_array_aux"))


def combine_leaf_predicate(base_fn, is_leaf):
    @ft.wraps(base_fn)
    def new_fn(*args, new_is_leaf=None):
        if new_is_leaf is None:
            combined_is_leaf = is_leaf
        else:

            def combined_is_leaf(arg):
                return is_leaf(arg) or new_is_leaf(arg)

        return base_fn(*args, is_leaf=combined_is_leaf)

    return new_fn


def leaf_predicate(x):
    return isinstance(x, ImplicitArray)


tree_map_with_implicit = combine_leaf_predicate(
    tu.tree_map,
    leaf_predicate,
)
tree_map_with_path_with_implicit = combine_leaf_predicate(
    tu.tree_map_with_path,
    leaf_predicate,
)
tree_flatten_with_implicit = combine_leaf_predicate(
    tu.tree_flatten,
    leaf_predicate,
)
tree_flatten_with_path_with_implicit = combine_leaf_predicate(
    tu.tree_flatten_with_path,
    leaf_predicate,
)
tree_leaves_with_implicit = combine_leaf_predicate(
    tu.tree_leaves,
    leaf_predicate,
)
tree_structure_with_implicit = combine_leaf_predicate(
    tu.tree_structure,
    leaf_predicate,
)


def flatten_one_implicit_layer(tree):
    def is_leaf_below_node(node, x):
        return isinstance(x, ImplicitArray) and x is not node

    def replace_subtree_implicits(node):
        return tu.tree_map(
            lambda _: 1,
            node,
            is_leaf=ft.partial(is_leaf_below_node, node),
        )

    prototype = tree_map_with_implicit(replace_subtree_implicits, tree)
    struct = tu.tree_structure(prototype)

    leaves = tree_leaves_with_implicit(tree)
    leaves = list(
        it.chain.from_iterable(
            (
                tu.tree_leaves(leaf, is_leaf=ft.partial(is_leaf_below_node, leaf))
                if isinstance(leaf, ImplicitArray)
                else [leaf]
            )
            for leaf in leaves
        )
    )
    return leaves, struct


def implicit_depth(tree):
    leaves = tree_leaves_with_implicit(tree)
    depth = 0
    while True:
        next_leaves = []
        any_implicit = False
        for leaf in leaves:
            if not isinstance(leaf, ImplicitArray):
                continue
            any_implicit = True
            next_leaves.extend(flatten_one_implicit_layer(leaf)[0])

        if not any_implicit:
            return depth

        depth += 1
        leaves = next_leaves


def _map_leaves_with_implicit_path(f, leaves, is_leaf, path_prefix=()):
    mapped_leaves = []
    for idx, leaf in enumerate(leaves):
        path = (*path_prefix, idx)
        if not isinstance(leaf, ImplicitArray) or is_leaf(path, leaf):
            mapped_leaves.append(f(leaf))
            continue

        subtree, substruct = flatten_one_implicit_layer(leaf)
        mapped_subtree = _map_leaves_with_implicit_path(
            f,
            subtree,
            is_leaf=is_leaf,
            path_prefix=path,
        )
        mapped_leaves.append(tu.tree_unflatten(substruct, mapped_subtree))
    return mapped_leaves


_rules: dict[core.Primitive, plum.Function] = {}


def register(
    primitive: core.Primitive | str,
    *,
    precedence: int = 0,
) -> Callable[[CT], CT]:
    if isinstance(primitive, str):
        lac = getattr(jax.lax, f"{primitive}_p", None)
        if lac is None:
            raise ValueError(f"couldn't verify given string primitive {primitive}_p")
        primitive = lac

    def _register(rule: CT) -> CT:
        try:
            existing_rule = _rules[primitive]
        except KeyError:

            def existing_rule():
                raise AssertionError()

            existing_rule.__name__ = f"{primitive}_dispatcher"
            existing_rule.__qualname__ = f"{primitive}_dispatcher"
            existing_rule = plum.Dispatcher().abstract(existing_rule)

            _rules[primitive] = existing_rule
        existing_rule.dispatch(rule, precedence=precedence)
        return rule

    return _register


def _default_process(
    primitive: core.Primitive,
    values: Sequence[chex.Array | ImplicitArray],
    params,
):
    arrays: list[chex.Array] = []
    for x in values:
        if _is_value(x):
            arrays.append(x.materialize())
        elif is_array(x):
            arrays.append(tp.cast(chex.Array, x))
        else:
            arrays.append(x)

    subfuns, bind_params = primitive.get_bind_params(params)
    return primitive.bind(*subfuns, *arrays, **bind_params)


def _wrap_tracer(x, trace: _CustomTrace):
    if _is_value(x):
        return _CustomTracer(trace, x)
    else:
        return x


def _unwrap_tracer(x, trace):
    if is_array(x):
        x = trace.full_raise(x)
    if isinstance(x, _CustomTracer):
        return x.value
    else:
        return x


class _CustomTracer(core.Tracer):
    __slots__ = ("value",)

    def __init__(self, trace: _CustomTrace, value: ImplicitArray) -> None:
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        return self.value.aval

    def full_lower(self):
        if isinstance(self.value, ImplicitArray):
            return self
        else:
            return core.full_lower(self.value)


class _CustomTrace(core.Trace[_CustomTracer]):
    def __init__(self, parent_trace, tag):
        super().__init__()
        self.tag = tag
        self.parent_trace = parent_trace

    def to_value(self, val):
        if isinstance(val, _CustomTracer) and val._trace.tag is self.tag:
            return val.value
        return val

    def process_primitive(self, primitive, tracers, params):
        values = [self.to_value(t) for t in tracers]
        values = tuple(values)
        implicit_idx = next(
            (i for i, v in enumerate(values) if isinstance(v, ImplicitArray)),
            None,
        )
        implicit_name = None
        if implicit_idx is not None:
            implicit_name = values[implicit_idx].__class__.__name__
            try:
                rule = _rules[primitive]
            except KeyError:
                with core.set_current_trace(self.parent_trace):
                    if WARN_ON_MATTER and implicit_name is not None:
                        warnings.warn(
                            f"No Custom Primitive been found for {primitive} (materializing {implicit_name})",
                            stacklevel=1,
                        )
                    out = _default_process(primitive, values, params)
            else:
                include_prim = False
                with core.set_current_trace(self.parent_trace):
                    try:
                        try:
                            method, _ = rule.resolve_method(values)
                        except (plum.NotFoundLookupError, plum.AmbiguousLookupError):
                            inhint = (primitive, *tuple(values))
                            include_prim = True
                            method, _ = rule.resolve_method(inhint)
                    except (plum.NotFoundLookupError, plum.AmbiguousLookupError):
                        if WARN_ON_MATTER and implicit_name is not None:
                            warnings.warn(
                                f"No Custom Primitive could match for {primitive} (materializing {implicit_name})",
                                stacklevel=1,
                            )
                        out = _default_process(primitive, values, params)
                    else:
                        if include_prim:
                            values = (primitive, *tuple(values))
                        out = method(*values, **params)
        else:
            with core.set_current_trace(self.parent_trace):
                subfuns, bind_params = primitive.get_bind_params(params)
                out = primitive.bind(*subfuns, *values, **bind_params)
        if primitive.multiple_results:
            out = [_CustomTracer(self, x) for x in out]
        else:
            out = _CustomTracer(self, out)
        return out

    def process_shard_map(self: _CustomTrace, primitive, fun, tracers, **params):
        tracers = [(arr.materialize() if _is_value(arr) else arr) for arr in [self.to_value(t) for t in tracers]]
        out = primitive.bind_with_trace(self.parent_trace, (fun, *tracers), params)
        if primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_map(self, map_primitive, f, tracers, **params):
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            out = _default_process(map_primitive, in_values, params)
        if map_primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_custom_transpose(self, prim, call, tracers, **params):
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            out = _default_process(prim, in_values, params)
        if prim.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_call(self, call_primitive, f, tracers, params):
        in_values = [self.to_value(t) for t in tracers]
        with core.set_current_trace(self.parent_trace):
            out = _default_process(call_primitive, in_values, params)
        if call_primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out]
        else:
            return _CustomTracer(self, out)

    def process_custom_jvp_call(
        self,
        primitive,
        fun,
        fwd,
        bwd,
        tracers,
        out_trees,
        symbolic_zeros,
    ):
        del fwd, bwd, out_trees, symbolic_zeros
        in_values = [self.to_value(t) for t in tracers]
        arrays: list[chex.Array] = []
        for x in in_values:
            if _is_value(x):
                arrays.append(x.materialize())
            elif is_array(x):
                arrays.append(tp.cast(chex.Array, x))
            else:
                arrays.append(x)
        with jax.core.set_current_trace(self.parent_trace):
            out_leaves = fun.call_wrapped(*arrays)
        if primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out_leaves]
        else:
            return _CustomTracer(self, out_leaves)

    def process_custom_vjp_call(
        self,
        primitive,
        fun,
        fwd,
        bwd,
        tracers,
        out_trees,
        symbolic_zeros,
    ):
        del fwd, bwd, out_trees, symbolic_zeros
        in_values = [self.to_value(t) for t in tracers]
        arrays: list[chex.Array] = []
        for x in in_values:
            if _is_value(x):
                arrays.append(x.materialize())
            elif is_array(x):
                arrays.append(tp.cast(chex.Array, x))
            else:
                arrays.append(x)
        with jax.core.set_current_trace(self.parent_trace):
            out_leaves = fun.call_wrapped(*arrays)
        if primitive.multiple_results:
            return [_CustomTracer(self, x) for x in out_leaves]
        else:
            return _CustomTracer(self, out_leaves)


def _custom_vjp_fwd_wrap(fwd, tag, in_treedef):
    def wrapped(*args):
        inputs = args[-len(in_treedef.children()) :]
        inputs = tu.tree_unflatten(in_treedef, inputs)
        out = fwd(*inputs)
        if not isinstance(out, tuple):
            out = (out,)
        out_flat, _ = tu.tree_flatten(out)
        return out_flat

    return wrapped


def _custom_vjp_bwd_wrap(bwd, tag, in_treedef):
    def wrapped(*args):
        out = bwd(*args)
        if not isinstance(out, tuple):
            out = (out,)
        out_flat, _ = tu.tree_flatten(out)
        return out_flat

    return wrapped


def _custom_vjp_fwd_wrap(fwd, tag, in_treedef):
    def wrapped(*args):
        inputs = tu.tree_unflatten(in_treedef, args)
        out = fwd(*inputs)
        if not isinstance(out, tuple):
            out = (out,)
        out_flat, out_tree = tu.tree_flatten(out)
        return out_flat, out_tree

    return wrapped, None


def _custom_vjp_bwd_wrap(bwd, tag, in_treedef):
    def wrapped(*args):
        res_and_cts = tu.tree_unflatten(in_treedef, args)
        out = bwd(*res_and_cts)
        if not isinstance(out, tuple):
            out = (out,)
        out_flat, out_tree = tu.tree_flatten(out)
        return out_flat, out_tree

    return wrapped, None


@lu.transformation_with_aux
def _custom_jvp_fun_wrap(tag, in_treedef, *in_leaves):
    in_values = tu.tree_unflatten(in_treedef, in_leaves)
    with core.take_current_trace() as parent_trace:
        trace = _CustomTrace(parent_trace, tag)
        in_tracers = [x if type(x) is SZ else _CustomTracer(trace, x) for x in in_values]
        with core.set_current_trace(trace):
            out_tracers = yield in_tracers, {}
            out_tracers = [jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t for t in out_tracers]
            out_values = [trace.to_value(t) for t in out_tracers]
            del out_tracers
        del trace, in_tracers
    out_leaves, out_treedef = tu.tree_flatten(out_values)
    yield out_leaves, out_treedef


@lu.transformation_with_aux
def _custom_jvp_jvp_wrap(tag, in_treedef, *in_primals_and_tangents):
    in_primals = in_primals_and_tangents[: len(in_primals_and_tangents) // 2]
    in_tangents = in_primals_and_tangents[len(in_primals_and_tangents) // 2 :]
    in_primal_values = tu.tree_unflatten(in_treedef, in_primals)
    in_tangent_values = tu.tree_unflatten(in_treedef, in_tangents)
    with core.take_current_trace() as parent_trace:
        trace = _CustomTrace(parent_trace, tag)
        in_tracers = [_CustomTracer(trace, x) for x in it.chain(in_primal_values, in_tangent_values)]
        with core.set_current_trace(trace):
            out_tracers = yield in_tracers, {}
            out_tracers = [jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t for t in out_tracers]
            out_values = [trace.to_value(t) for t in out_tracers]
            out_primal_values = out_values[: len(out_values) // 2]
            out_tangent_values = out_values[len(out_values) // 2 :]
            out_primal_values2 = []
            out_tangent_values2 = []
            assert len(out_primal_values) == len(out_tangent_values)
            for primal, tangent in zip(out_primal_values, out_tangent_values):  # noqa
                if primal.__class__ != tangent.__class__:
                    primal = primal.materialize()
                    tangent = tangent.materialize()
                out_primal_values2.append(primal)
                out_tangent_values2.append(tangent)
            del out_tracers
        del trace, in_tracers
    out_primals, out_primal_treedef = tu.tree_flatten(out_primal_values2)
    out_tangents, out_tangent_treedef = tu.tree_flatten(out_tangent_values2)
    if out_primal_treedef != out_tangent_treedef:
        raise ValueError("Primals and tangents had the same class, but different flattened results.")
    yield out_primals + out_tangents, out_primal_treedef


def _is_value(x) -> TypeGuard[ImplicitArray]:
    return isinstance(x, ImplicitArray)


@register(jax._src.pjit.jit_p)
def _(
    *args: ImplicitArray | ArrayLike,
    jaxpr,
    inline,
    **kwargs,
):
    del kwargs
    fun = use_implicit(jax.extend.core.jaxpr_as_fun(jaxpr))
    return fun(*args)
    if inline:
        return fun(*args)
    else:
        leaves, treedef = tu.tree_flatten(args)
        flat_fun = lambda x: fun(*tu.tree_unflatten(treedef, x))  # noqa
        return jax.jit(flat_fun)(leaves)


_sentinel = object()


@register(jax.lax.while_p)
def _(
    *args: ImplicitArray | ArrayLike,
    cond_nconsts: int,
    cond_jaxpr,
    body_nconsts: int,
    body_jaxpr,
):
    cond_consts = args[:cond_nconsts]
    body_consts = args[cond_nconsts : cond_nconsts + body_nconsts]
    init_vals = args[cond_nconsts + body_nconsts :]

    quax_cond_fn = implicit(core.jaxpr_as_fun(cond_jaxpr))
    quax_cond_jaxpr = jax.make_jaxpr(quax_cond_fn)(*cond_consts, *init_vals)
    quax_body_fn = implicit(core.jaxpr_as_fun(body_jaxpr))
    quax_body_jaxpr = jax.make_jaxpr(quax_body_fn)(*body_consts, *init_vals)

    cond_leaves, _ = tu.tree_flatten(cond_consts)
    body_leaves, _ = tu.tree_flatten(body_consts)
    init_val_leaves, val_treedef = tu.tree_flatten(init_vals)
    try:
        out_val = jax.lax.while_p.bind(
            *cond_leaves,
            *body_leaves,
            *init_val_leaves,
            cond_nconsts=cond_nconsts,
            cond_jaxpr=quax_cond_jaxpr,
            body_nconsts=body_nconsts,
            body_jaxpr=quax_body_jaxpr,
        )
    except Exception as e:
        raise RuntimeError("You should customize while prim for your usecase") from e
    result = tu.tree_unflatten(val_treedef, out_val)
    return result


@register("cond")
def _(
    index: chex.Array,
    *args: tp.Any,
    branches: tuple,
    linear=_sentinel,
):
    flat_args, in_tree = tu.tree_flatten(args)

    out_trees = []
    _branches = []
    for jaxpr in branches:

        def flat__call(flat_args):
            args = tu.tree_unflatten(in_tree, flat_args)
            out = use_implicit(core.jaxpr_as_fun(jaxpr))(*args)  # noqa
            flat_out, out_tree = tu.tree_flatten(out)
            out_trees.append(out_tree)
            return flat_out

        _jaxpr = jax.make_jaxpr(flat__call)(flat_args)
        _branches.append(_jaxpr)

    if tp.Any(tree_outs_i != out_trees[0] for tree_outs_i in out_trees[1:]):
        raise TypeError("all branches output must have the same pytree.")

    if linear is _sentinel:
        maybe_linear = {}
    else:
        maybe_linear = dict(linear=linear)
    out_val = jax.lax.cond_p.bind(index, *flat_args, branches=tuple(_branches), **maybe_linear)
    result = tu.tree_unflatten(out_trees[0], out_val)
    return result
