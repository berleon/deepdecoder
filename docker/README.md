# Dockerfile

This docker file holds all dependencies to run the RenderGAN code.
There is a general [Dockerfile](Dockerfile) for all the dependencies.
And the `./generate_user_dockerfile.sh`  to create a Dockerfile to match the user id of the host user.

## Simple setup

```
make all
```

## Setup

Build the base image with:

```
make base-image
```

## Customize

To match the user id of the host system, create your own custom docker file with:

```
$ ./generate_user_dockerfile.sh
$ cd rendergan-$(USER)
$ docker build --tag rendergan-$(USER)
```

See `./generate_user_dockerfile.sh --help` for howto set username, uid or password.
The Dockerfile is created in a new directory `rendergan-$USER`. You can build it with
`docker build --tag [your image name] .`.
Sometimes, some cuda libaries are not found then just run `ldconfig` as root.

