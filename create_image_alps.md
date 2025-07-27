Assuming a partition X

1. Get a compute node:

srun -t 5:00:00 -A a-infra02 --container-writable --pty bash

2. Clone the sailor repo

git clone https://github.com/eth-easl/sailor.git
cd sailor

3. While in the folder with the Dockerfile, create a new a new image (adjust the name as you want)
podman build -t test:v1 .

4. You can see your image now using this command
podman images

5. use enroot to export the image into a squash file
enroot import -o test.sqsh podman://test:v1

6. Make it readable
setfacl -b test.sqsh
chmod 755 test.sqsh

7. Create a .toml file

8. To get a container with the image running and get a shell
