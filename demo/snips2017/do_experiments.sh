#!/bin/bash

set -e

err(){
    echo "E: $*" >>/dev/stderr
}

if [ "$(ls -l . | grep demo.py | wc -l)" -eq 0 ]; then
	if [ "$(ls -l ./demo/snips2017 | grep demo.py | wc -l)" -eq 0 ]; then
		err "$PWD is not a directory with SNIPS-2017 demo!"
        	exit 1
	fi
else
	cd ../..
fi

echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 50, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type usual 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 100, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type usual 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 200, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv200_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 200 --conv2 200 --conv3 200 --conv4 200 --conv5 200 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type usual 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 200, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv200_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 200 --conv2 200 --conv3 200 --conv4 200 --conv5 200 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 200, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv200_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 200 --conv2 200 --conv3 200 --conv4 200 --conv5 200 --hidden 0 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian 
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 50, hidden layer is 100"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_hid100.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 100 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, hidden layer is 100"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_hid100.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 100 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, hidden layer is 100"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_hid100.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 100 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 100, hidden layer is 300"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_hid300.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 300 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, hidden layer is 300"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_hid300.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 300 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, hidden layer is 300"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_hid300.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 300 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 200, hidden layer is 500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv200_hid500.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 200 --conv2 200 --conv3 200 --conv4 200 --conv5 200 --hidden 500 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 200, hidden layer is 500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv200_hid500.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 200 --conv2 200 --conv3 200 --conv4 200 --conv5 200 --hidden 500 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 200, hidden layer is 500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv200_hid500.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 200 --conv2 200 --conv3 200 --conv4 200 --conv5 200 --hidden 500 --num_monte_carlo 10 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian


