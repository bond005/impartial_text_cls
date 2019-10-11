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
echo "Usual network, number of convolution filters for each kernel size is 100, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 0 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 100, hidden layer is 1000"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_hid1000.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1000 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 100, hidden layer is 1500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_hid900.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1500 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 100, hidden layer is 1000, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_hid1000_1000.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1000:2 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network, number of convolution filters for each kernel size is 100, hidden layer is 1500, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_conv100_hid1500_1500.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1500:2 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type usual
echo "===================="

echo "===================="
echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 0 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, hidden layer is 1000"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_hid1000.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1000 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, hidden layer is 1500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_hid900.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1500 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, hidden layer is 1000, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_hid1000_1000.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1000:2 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Bayesian network, number of convolution filters for each kernel size is 100, hidden layer is 1500, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/bayesian_nn_conv100_hid1500_1500.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1500:2 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type bayesian
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, without hidden layer"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_nohid.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 0 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, hidden layer is 1000"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_hid1000.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1000 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, hidden layer is 1500"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_hid900.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1500 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, hidden layer is 1000, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_hid1000_1000.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1000:2 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="

echo ""
echo ""
echo "===================="
echo "Usual network with additional class, number of convolution filters for each kernel size is 100, hidden layer is 1500, two hidden layers"
echo "===================="
echo ""
PYTHONPATH=$PWD python -u demo/snips2017/demo.py -m demo/snips2017/data/usual_nn_with_additional_class_conv100_hid1500_1500.pkl -d /home/ubuntu/sources/nlu-benchmark/2017-06-custom-intent-engines/ --conv1 100 --conv2 100 --conv3 100 --conv4 100 --conv5 100 --hidden 1500:2 --num_monte_carlo 100 --batch_size 64 --gpu_frac 0.95 --nn_type additional_class
echo "===================="