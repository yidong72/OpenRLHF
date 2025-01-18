docker run --gpus all -it --rm --shm-size=128g \
	      --env USER_ID=$(id -u) --env GROUP_ID=$(id -g)  \
	            -v $PWD:/openrlhf -v $PWD/../../:/home/myuser -v $PWD/../data/:/datasets -v $PWD/../results/:/results -v $PWD/../:/Projects --ulimit memlock=-1 --ulimit \
		                    stack=67108864 openrlhf
