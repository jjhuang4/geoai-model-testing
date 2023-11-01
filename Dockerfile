FROM jupyter/minimal-notebook:2022-01-12

WORKDIR /geoai-model-testing

ADD . /geoai-model-testing

RUN conda install -c conda-forge pytorch  pycocotools torchvision pandas numpy matplotlib geopandas cudatoolkit
#python -m pip install git+https://github.com/jjhuang4/geoai-model-testing@6643034e401eb5c1253d4d909e771985d3e5fd42
#pip install --no-cache-dir -r requirements.txt
#RUN python -m pip install git+https://github.com/pysal/access@c2fc9a1a6c92a223ebe3157c1b4bc248eacc691f

#EXPOSE

#CMD