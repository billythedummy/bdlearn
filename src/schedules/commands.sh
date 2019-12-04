OUT_PATH='../../include/schedules'

../../bin/batchim2col -g batchim2col_schedule_gen -o $OUT_PATH \
    -e static_library,h,schedule,cpp -f libbatchim2col \
    target=host auto_schedule=true machine_params=32,16777216,40