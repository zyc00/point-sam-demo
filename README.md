# point-sam-demo

## Install
Install PyTorch by following [this](https://pytorch.org).

Then install frontend and backend dependences as well as Point-SAM.
```
pip install flask
pip install flask-cors
pip install safetensors
pip install git+https://github.com/zyc00/Point-SAM.git
```

## Run Demo
```
python app.py --host [your_host] --port [your_port]
```

Then you can use the demo in your browser. Some example meshes for the demo are provided in [examples](./examples/). If you want to use your own mesh, make sure the number of faces is less then 500k. We suggest using meshes with less than 100k faces.