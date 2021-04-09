import argparse

parser = argparse.ArgumentParser(description = 'model')

from os.path import join

root_path = '.' # source folder path

saving_path = join(root_path, 'results')
loading_path = join(root_path, 'results') # load model same posenet_finetune_pathas saving_path

#filepath_SHD = join(root_path, join('data', 'training_SHD.h5'))
#filepath_RHD = join(root_path, join('data', 'training_RHD.h5'))
filepath_RHD = '/data/HPE_DATA/h5/training_RHD_20200525.h5'
filepath_SHD = '/data/HPE_DATA/h5/training_SHD_20200525.h5'

test_model_path = join(root_path, 'results/checkpoint/net.t7')

#filepath_SHD_test = join(root_path, join('data', 'test_SHD.h5'))
filepath_SHD_test = '/data/HPE_DATA/h5/evaluation_SHD_20200525.h5'

# only to create h5
filepath_SHD_raw = '/media/armagan/2TB/work/data/stereohandtracking'
filepath_RHD_raw = '/media/armagan/My Passport/work/data/RHD_published_v2'

posenet_finetune_path = join(root_path, join('data', 'posenet3d-rhd-stb-slr-finetuned.pickle'))
mano_path = join(root_path, join('data', 'MANO_LEFT.pkl'))

## PATHS
parser.add_argument(
    '--root-path',
    default = root_path,
    type = str,
    help = 'path to the source code dir.'
)


parser.add_argument(
    '--posenet-finetune-path',
    default = posenet_finetune_path,
    type = str,
    help = ''
)


parser.add_argument(
    '--mano-path',
    default = mano_path,
    type = str,
    help = ''
)

parser.add_argument(
    '--saving-path',
    default = saving_path,
    type = str,
    help = ''
)
parser.add_argument(
    '--loading-path',
    default = loading_path,
    type = str,
    help = ''
)

parser.add_argument(
    '--filepath-SHD',
    default = filepath_SHD,
    type = str,
    help = ''
)
parser.add_argument(
    '--filepath-SHD-raw',
    default = filepath_SHD_raw,
    type = str,
    help = ''
)
parser.add_argument(
    '--filepath-RHD',
    default = filepath_RHD,
    type = str,
    help = ''
)
parser.add_argument(
    '--filepath-RHD-raw',
    default = filepath_RHD_raw,
    type = str,
    help = ''
)

# network params
encoder_feature_count = {
    'resnet50' : 2048
}

crop_size = {
    'resnet50':96
}

parser.add_argument(
    '--input-size',
    type = int,
    default = 224,
    help = 'the count of theta param'
)

parser.add_argument(
    '--encoder-network',
    type = str,
    default = 'resnet50',
    help = 'the encoder network name'
)

parser.add_argument(
    '--total-theta-count',
    type = int,
    default = 63,
    help = 'the count of theta param'
)

parser.add_argument(
    '--num-joints',
    type = int,
    default = 21,
    help = 'number of hand joints to estimate'
)

#############################
# model TRAINING paramaters #
#############################
parser.add_argument(
    '--epoch',
    type = int,
    default = 40,
    help = 'default epoch number'
)

parser.add_argument(
    '--batch-size1',
    type = int,
    default = 10,
    help = 'batch size for RHD train'
)

parser.add_argument(
    '--batch-size2',
    type = int,
    default = 10,
    help = 'batch size for SHD train'
)


#############################
# model TEST parameters
#############################
parser.add_argument(
    '--test-model-path',
    type = str,
    default = test_model_path,
    help = 'load model path to test'
)

parser.add_argument(
    '--filepath-SHD-test',
    default = filepath_SHD_test,
    type = str,
    help = ''
)

parser.add_argument(
    '--batch-size-eval',
    type = int,
    default = 1,
    help = 'batch size for SHD test'
)




args = parser.parse_args()
encoder_network = args.encoder_network
args.feature_count = encoder_feature_count[encoder_network]
args.crop_size = crop_size[encoder_network]
