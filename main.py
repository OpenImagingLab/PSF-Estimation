from model.PSF_mlp import *
from model.optics_rgb import *
import utils.train as train

if __name__ == '__main__':
    # default 1
    num = 0
    torch.manual_seed(num)
    source = './configs/63762BB.yaml'
    print(f'seed = {num}')
    # print(source)
    args = train.config(source)

    result_path = args['result_path']
    IS = IS(filepath=args['in_path'])
    IS.seidel_basis = IS.s_basis(IS.wf_res, type=args['net'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = PSF_mlp().to(device)

    shiftnet = shift_net()
    train.train(net, shiftnet, IS, args)