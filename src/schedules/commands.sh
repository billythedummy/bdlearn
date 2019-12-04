OUT_PATH='../../include/schedules'

../../bin/batchim2col -g batchim2col_schedule_gen -o $OUT_PATH \
    -e static_library,h,schedule -f libbatchim2col \
    target=host auto_schedule=true machine_params=32,16777216,40

../../bin/batchcol2imaccum -g batchcol2imaccum_schedule_gen -o $OUT_PATH \
    -e static_library,h,schedule -f libbatchcol2imaccum \
    target=host auto_schedule=true machine_params=32,16777216,40

../../bin/batchmatmulabr -g batchmatmulabr_schedule_gen -o $OUT_PATH \
    -e static_library,h,schedule -f libbatchmatmulabr \
    target=host auto_schedule=true machine_params=32,16777216,40

    