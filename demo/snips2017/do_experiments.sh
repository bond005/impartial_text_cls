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
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_nohid.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 0 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 50, hidden layer is 500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_hid500.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 500 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 50, hidden layer is 900"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_hid900.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 900 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 50, hidden layer is 500, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_hid500_500.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 500:2 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 50, hidden layer is 900, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv050_hid900_900.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 900:2 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo "===================="
echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_nohid.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 0 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, hidden layer is 500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_hid500.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 500 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, hidden layer is 900"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_hid900.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 900 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, hidden layer is 500, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_hid500_500.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 500:2 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 50, hidden layer is 900, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv050_hid900_900.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 900:2 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_nohid.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 0 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, hidden layer is 500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_hid500.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 500 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, hidden layer is 900"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_hid900.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 900 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, hidden layer is 500, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_hid500_500.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 500:2 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 50, hidden layer is 900, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv050_hid900_900.pkl -d "$1" --conv1 50 --conv2 50 --conv3 50 --conv4 50 --conv5 50 --hidden 900:2 --num_monte_carlo 20 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="