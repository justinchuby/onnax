Python API
==========

The functional transforms preserve the input container type:

.. list-table::
   :header-rows: 1

   * - Input
     - Return
   * - ``onnx_ir.Model``
     - ``onnx_ir.Model``
   * - ``onnx_ir.Graph``
     - ``onnx_ir.Graph``
   * - ``onnx_ir.GraphView``
     - independent ``onnx_ir.Graph``
   * - ``onnx.ModelProto``
     - ``onnx.ModelProto``

Graphs and views carry their own opset imports. Wrap a returned graph in an
``onnx_ir.Model`` to export a standalone model. The compatible pass classes
continue to accept IR models and return ``PassResult``.

.. automodule:: onnax
   :members:
   :undoc-members:
