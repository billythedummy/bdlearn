# Dependencies

## Already Packaged
- [libpopcnt 2.2](https://github.com/kimwalisch/libpopcnt)

## Not Packaged
- [Halide](https://github.com/halide/Halide) - libhalide.so should be in LD_LIBRARY_PATH

# Acknowledgements
- [kimwalisch](https://github.com/kimwalisch) for libpopcnt

# For Ryan
- Halide reads from left to right in increasing major: Any Func/Expr(x, y, z) means the strides through the buffer are x changes fastest followed by y, followed by z. x stride = 1, y stride = x, z stride = y*x
- To match this with our row_major storage, we have to do Func/Expr(cols, rows, channels)