FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /app

RUN set -eux && \
    apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

RUN apt-get update && apt-get -y install sudo
RUN sudo su -

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

RUN apt-get -y install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev gdb
RUN pip uninstall xgboost transformer_engine flash_attn pynvml -y
RUN pip install vllm==0.7.2
RUN pip install accelerate bitsandbytes datasets deepspeed==0.16.3 einops flash-attn==2.7.0.post2 isort jsonlines loralib optimum packaging peft ray[default]==2.42.0 tensorboard torch torchmetrics tqdm transformers==4.48.2 transformers_stream_generator wandb wheel
# RUN pip install celery redis datasets=='2.16.1' fire pre-commit
# RUN pip install hydra-core ipython pudb nvidia-pytriton
# RUN pip install git+https://github.com/Kipok/NeMo-Skills.git
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh
WORKDIR /workspace
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/bin/bash"]
