FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PATH="/opt/conda/bin:$PATH"

# Copy requirements from the notebook folder
COPY requirements.txt /app/requirements.txt


# Install system build deps needed by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#Install conda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
conda install -n base -c conda-forge mamba -y 

# Install 2 env (Retrieval,Generation)
RUN mamba create -n retenv python=3.9 -c anaconda -c conda-forge mkl_fft==1.3.0 mkl_random==1.2.1 mkl-service==2.3.0 -y && \
mamba create -n genenv python=3.9 -c anaconda -c conda-forge mkl_fft==1.3.0 mkl_random==1.2.1 mkl-service==2.3.0 -y

#Clone docprompt
RUN git clone https://github.com/shuyanzhou/docprompting

# Install dependencies for both envs
RUN conda run -n retenv pip install pip==23.3.2 && \
conda run -n retenv pip install "setuptools<74" && \
conda run -n genenv pip install pip==23.3.2 && \
conda run -n genenv pip install "setuptools<74" 

RUN conda run -n retenv pip install git+https://github.com/anaconda/anaconda-project.git@v0.9.1 && \
conda run -n retenv pip install git+https://github.com/anaconda/anaconda-client@1.7.2 && \
conda run -n genenv pip install git+https://github.com/anaconda/anaconda-project.git@v0.9.1 && \
conda run -n genenv pip install git+https://github.com/anaconda/anaconda-client@1.7.2

RUN conda run -n retenv pip  install extension-helpers && \
conda run -n retenv pip  install numpy==1.20.1  && \
conda run -n retenv pip install astropy==4.2.1 --no-build-isolation  && \
conda run -n genenv pip  install extension-helpers && \
conda run -n genenv pip  install numpy==1.20.1  && \
conda run -n genenv pip install astropy==4.2.1 --no-build-isolation 

RUN conda run -n retenv pip install bert-score==0.3.11 && \
conda run -n retenv pip install git+https://github.com/mkleehammer/pyodbc@4.0.0 && \
conda run -n genenv pip install bert-score==0.3.11 && \
conda run -n genenv pip install git+https://github.com/mkleehammer/pyodbc@4.0.0

RUN conda run -n retenv pip install git+https://github.com/anaconda-graveyard/clyent@1.2.2 && \
conda run -n retenv pip install bottleneck==1.3.2 --no-build-isolation && \
conda run -n genenv pip install git+https://github.com/anaconda-graveyard/clyent@1.2.2 && \
conda run -n genenv pip install bottleneck==1.3.2 --no-build-isolation 

RUN conda run -n retenv pip install cython==0.29.23 && \
conda run -n genenv pip install cython==0.29.23

RUN conda run -n retenv pip install simplegeneric && \
conda run -n retenv pip install singledispatch && \
conda run -n retenv pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
conda run -n genenv pip install simplegeneric && \
conda run -n genenv pip install singledispatch && \
conda run -n genenv pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN conda run -n retenv pip install torchaudio==0.7.2 && \
conda run -n retenv pip install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
conda run -n genenv pip install torchaudio==0.7.2 && \
conda run -n genenv pip install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN conda install -n retenv -c conda -c conda-forge  h5py==2.10.0 && \
conda install -n genenv -c conda -c conda-forge  h5py==2.10.0

RUN conda install -n retenv  -c conda-forge  gmpy2==2.1.0b1 && \
conda install -n genenv  -c conda-forge  gmpy2==2.1.0b1

RUN conda run -n retenv pip install pyflakes==2.3.0 && \
conda run -n genenv pip install pyflakes==2.3.0

RUN conda run -n retenv pip install -r  /app/requirements.txt --no-deps && \
conda run -n genenv pip install -r  /app/requirements.txt --no-deps

#Retrieval
RUN conda install -n retenv  -c pytorch  pytorch>=2.1.0 && \
conda run -n retenv  pip install sacrebleu && \
conda run -n retenv  pip install -U transformers==4.31.0 huggingface_hub  sentencepiece tokenizers

#cd docprompting &&HUGGINGFACE_HUB_URL=https://huggingface.co PYTHONPATH=$PYTHONPATH:'/content/docprompting' conda run -n retenv python retriever/simcse/run_inference.py \
#  --model_name "neulab/docprompting-codet5-python-doc-retriever" \
#  --source_file data/conala/conala_nl.txt \
#  --target_file data/conala/python_manual_firstpara.tok.txt \
#  --source_embed_save_file data/conala/.tmp/src_embedding \
#  --target_embed_save_file data/conala/.tmp/tgt_embedding \
#  --sim_func cls_distance.cosine \
#  --num_layers 12 \
#  --save_file data/conala/retrieval_results.json

#Generation
RUN conda run -n genenv  pip install sacrebleu && \
conda run -n genenv pip install -U torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
conda run -n genenv pip install -U torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html && \
conda run -n genenv pip install -U torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
conda run -n genenv  pip install -U transformers==4.2.1 huggingface_hub==0.0.12  sentencepiece tokenizers==0.9.4 && 

COPY model.py /app/docprompting/generator/fid/src/model.py

#!cd docprompting && PYTHONPATH=$PYTHONPATH:'/content/docprompting' conda run -n genenv  python generator/fid/test_reader_simple.py \
#    --model_path /content/docprompting/models/generator/conala.fid.codet5.top10/checkpoint/best_dev/best_dev \
#    --tokenizer_name models/generator/codet5-base \
#    --eval_data data/conala/example_fid_data.json \
#    --per_gpu_batch_size 3 \
#    --n_context 10 \
#    --name conala.fid.codet5.top10 \
#    --checkpoint_dir models/generator  \
#    --result_tag test_same \
#    --main_port 81692
CMD ["bash"]

