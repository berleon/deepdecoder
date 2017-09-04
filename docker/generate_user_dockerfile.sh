#! /usr/bin/env bash

if [ "$1" == "--help" ]; then
    echo "$0 generate user for the docker file"
    echo "    -u | --user NAME          name of the user. default: current user"
    echo "    --uid UID                 uid of the user. default: uid of current user"
    echo "    -p | --password PASSWORD  password of the user. default: username"
    exit 0
fi

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -u|--user)
    USER="$2"
    shift # past argument
    ;;
    --uid)
    UID="$2"
    shift # past argument
    ;;
    -p|--password)
    PASSWORD="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

echo $USER
echo $UID
echo $PASSWORD
if [ "$PASSWORD" == "" ]; then
    PASSWORD="$USER"
fi


OUT_DIR="rendergan-$USER"
mkdir -p  "$OUT_DIR"

cp keras.json $OUT_DIR/keras.json
cp theanorc $OUT_DIR/theanorc

cat <<EOF  > $OUT_DIR/Dockerfile
FROM rendergan
MAINTAINER github@leon-sixt.de

# setup sudo
# RUN groupadd --system sudo
# RUN echo "%sudo ALL=(ALL) ALL" >> /etc/sudoers

# Use the same gid and uid as your user on the host system. You can find them
# out with the `id`  programm. This way the file ownership in mapped directories is
# consistent with the host system.
#
RUN groupadd --gid $UID $USER
RUN useradd --uid $UID  --gid $USER \
    --home-dir /home/$USER --shell /usr/bin/zsh  \
    --groups sudo,$USER \
    --password $USER \
    $USER

# default password $USER
RUN echo $USER:$PASSWORD | chpasswd && \
    echo root:$PASSWORD | chpasswd

ENV HOME /home/$USER
WORKDIR $HOME
RUN chown -R $USER:$USER .
USER $USER

COPY keras.json $HOME/.keras/keras.json
COPY theanorc $HOME/.theanorc

EOF
