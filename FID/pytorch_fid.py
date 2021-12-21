    '''
    use the pytorch-fid to calculate the FID score.
    calculate with 10k images, in different 9900 epochs, separately
    '''
    
import subprocess
import shlex
# import json
import pprint

# PATH
PATH = "/home/xchen/GANs/DCGAN-PyTorch/samples/1101_mnist_10kepochs_hatano/"
FILE_NAME = "1101_mnist_10kepochs_hatano"
fid = 'python3 -m pytorch_fid'

dict = {}

# calc the pytorch_fid
for i in range(0, 9901, 100):

    real_path = ' ' + PATH + str(i) + '/real_images'
    fake_path = ' ' + PATH + str(i) + '/fake_images'

    command_line = fid + real_path + fake_path

    args = shlex.split(command_line)

    res = subprocess.run(args, shell=False, stdout=subprocess.PIPE, text=True)

    dict[i] = float(res.stdout[6:-1])


with open(FILE_NAME+'.log', "w") as tf:
    # tf.write(json.dumps(dict, sort_keys=True, indent=4))

    print(PATH + '\n', file=tf)

    pprint.pprint(sorted(dict.items(), key=lambda kv: kv[1]), stream=tf)
