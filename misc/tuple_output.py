"""
tuple_output.py

一个轻量级的工具，用于动态创建命名元组(namedtuple)实例，
方便封装模型中间输出，保持结构清晰且兼容ONNX导出。

用法示例：

    output = TupleOutput(features=feat_tensor, attention=attn_tensor)
    print(output.features)
"""

from collections import namedtuple
from typing import Any, Optional, Dict, Union


class TupleOutput:
    """
    轻量级命名元组封装工具。

    通过传入字段键值对，动态创建对应的 namedtuple 实例。

    例子：
        output = TupleOutput(features=..., attention=...)
    """

    def __new__(
        cls,
        *args: Union[Dict[str, Any], None],
        name: Optional[str] = "TupleOutput",
        **kwargs: Any,
    ):
        # 合并 dict 和 kwargs
        data = {}
        for arg in args:
            if arg is not None:
                if not isinstance(arg, dict):
                    raise TypeError(
                        f"Expected dict as positional argument, got {type(arg)}"
                    )
                data.update(arg)
        data.update(kwargs)

        if not data:
            raise ValueError("No data provided to TupleOutput.")

        StructType = namedtuple(name, data.keys())
        return StructType(**data)


if __name__ == "__main__":
    # 测试
    out = TupleOutput(features="feat", attention="attn")
    print(out)
    print(out.features, out.attention)

    d = {"foo": 123, "bar": "abc"}
    out2 = TupleOutput(d, name="MyOutput")
    print(out2)
    print(out2.foo, out2.bar)
