import torch
from .ccell_model import CCell
from . import networks, losses


class TCell(CCell):
    """This class implements the transformer for image completion"""
    def name(self):
        return "Transformer Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--down_layers', type=int, default=4, help='# times down sampling for refine generator')
        parser.add_argument('--mid_layers', type=int, default=6, help='# times middle layers for refine generator')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for discriminator loss')
            parser.add_argument('--lambda_lp', type=float, default=10.0, help='weight for the perceptual loss')
            parser.add_argument('--lambda_gradient', type=float, default=0.0, help='weight for the gradient penalty')

        return parser

    def __init__(self, opt):
        """inital the Transformer model"""
        CCell.__init__(self, opt)
        self.visual_names = ['img_A', 'img_M', 'img_N', 'img_g', 'img_msk']
        self.model_names = ['E', 'G', 'D', 'T']
        self.loss_names = ['G_rec', 'G_lp', 'G_GAN', 'D_real', 'D_fake']

        self.netE = networks.define_E(opt)
        self.netT = networks.define_T(opt)
        self.netG = networks.define_G(opt)
        self.netD = networks.define_D(opt, opt.fine_size)

        if self.isTrain:
            # define the loss function
            self.L1loss = torch.nn.L1Loss()
            self.GANloss = losses.GANLoss(opt.gan_mode).to(self.device)
            self.NormalVGG = losses.Normalization(self.device)
            self.LPIPSloss = losses.LPIPSLoss(ckpt_path=opt.lipip_path).to(self.device)
            if len(self.opt.gpu_ids) > 0:
                self.LPIPSloss = torch.nn.parallel.DataParallel(self.LPIPSloss, self.opt.gpu_ids)
            # define the optimizer
            self.optimizerG = torch.optim.Adam(list(self.netE.parameters()) + list(self.netG.parameters())
                                               + list(self.netT.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 4, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizerG)
            self.optimizers.append(self.optimizerD)
        else:
            self.visual_names = ['img_A', 'img_M', 'img_N']

    def set_input(self, input):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input

        self.image_paths = self.input['img_path']
        self.img_A = self.input['img_A'].to(self.device) * 2 - 1
        self.img_M = self.input['img_M'].to(self.device) * 2 - 1
        self.img_N = self.input['img_N'].to(self.device) * 2 - 1
        self.img_msk = (self.img_N.sum(dim=1, keepdims=True) > -2.0).type_as(self.img_N)

    def forward(self):
        """Run forward processing to get the outputs"""
        out = self.netE(self.img_A)
        out = self.netT(out)
        self.img_g = self.netG(out)