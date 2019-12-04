# Dependencies

## Packaged
- [darknet's](https://github.com/pjreddie/darknet) [stb_image](https://github.com/pjreddie/darknet/blob/master/src/stb_image.h)

## Not Packaged
- [Halide](https://github.com/halide/Halide) - libhalide.so should be in LD_LIBRARY_PATH

# For Ryan
- Halide reads from left to right in increasing major: Any Func/Expr(x, y, z) means the strides through the buffer are x changes fastest followed by y, followed by z. x stride = 1, y stride = x, z stride = y*x
- To match this with our row_major storage, we have to do Func/Expr(cols, rows, channels)

# Other notes:
- batchnorm only works well for larger batch sizes, not shit like 2