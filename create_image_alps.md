This guide assumes you already have successfully logged in to Alps, and added to the a-infra-02 project.

1. ssh into Clariden and go to the '/capstor/scratch/cscs/$USER' directory

2. Get a compute node: `srun -t 5:00:00 -A a-infra02 --container-writable --pty bash`

3. Clone the sailor repo `git clone https://github.com/eth-easl/sailor.git && cd sailor `

4. While in the folder with the Dockerfile, create a new a new image (adjust the name as you want)
`podman build -t test:v1 .`

5. You can see your image now using this command
`podman images`

6. use enroot to export the image into a squash file
`enroot import -o test.sqsh podman://test:v1`

7. Make it readable
`setfacl -b test.sqsh && chmod 755 test.sqsh`

8. Create a .toml file or just reuse the one in [clariden_scripts/sailor.toml](clariden_scripts/sailor.toml)

9. To get a container with the image running and get a shell

`srun  -t 5:00:00 -A a-infra02 --container-writable --environment=/capstor/scratch/cscs/$USER/sailor/test.toml  --pty bash`

10. Run a simple training job with just 1 GPU to check all works:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed/
export SAILOR_LOGS_DIR=logs
bash run.sh 1 0 127.0.0.1 1234 1 1 1 1 1
```
