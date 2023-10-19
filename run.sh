docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p0.0.0.0:4040:24 \
            --gpus all \
            --name retina \
            --volume /mnt/sdb1/arman_scripts:/workdir \
            parking_trt:latest
