import torch
from .base_model import BaseModel
from . import networks, losses


class CCell(BaseModel):
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
        BaseModel.__init__(self, opt)
        self.visual_names = ['img_A', 'img_M', 'img_N', 'img_g', 'img_msk']
        self.model_names = ['E', 'G', 'D',]
        self.loss_names = ['G_rec', 'G_lp', 'G_GAN', 'D_real', 'D_fake']

        self.netE = networks.define_E(opt)
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
            self.optimizerG = torch.optim.Adam(list(self.netE.parameters()) + list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
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
        self.img_msk = (self.img_N.sum(dim=1, keepdims=True) > -2.5).type_as(self.img_M)

    @torch.no_grad()
    def test(self):
        """Run forward processing for testing"""
        self.forward()
        self.save_results(self.img_g, path=self.opt.save_dir + '/img_out')

    def forward(self):
        """Run forward processing to get the outputs"""
        out = self.netE(self.img_A)
        self.img_g = self.netG(out)

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: discriminator loss
        """
        self.loss_D_real = self.GANloss(netD(real), True, is_dis=True)
        self.loss_D_fake = self.GANloss(netD(fake), False, is_dis=True)
        loss_D = self.loss_D_real + self.loss_D_fake
        if self.opt.lambda_gradient > 0:
            self.loss_D_Gradient, _ = losses.cal_gradient_penalty(netD, real, fake, real.device, lambda_gp=self.opt.lambda_gradient)
            loss_D += self.loss_D_Gradient
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate the GAN loss for discriminator"""
        real = self.img_M.detach()
        fake = self.img_g.detach()
        self.loss_D = self.backward_D_basic(self.netD, real, fake) if self.opt.lambda_g > 0 else 0

    def backward_G(self):
        """Calculate the loss for generator"""
        self.loss_G_GAN = 0
        self.loss_G_rec = 0
        self.loss_G_lp =0
        fake = self.img_g
        self.loss_G_GAN += self.GANloss(self.netD(fake), True) * self.opt.lambda_g if self.opt.lambda_g > 0 else 0
        self.loss_G_rec += (self.L1loss(self.img_M*self.img_msk, self.img_g*self.img_msk) * 10 +
                            self.L1loss(self.img_M, self.img_g)) * self.opt.lambda_rec
        norm_real = self.NormalVGG((self.img_M + 1) * 0.5)
        norm_fake = self.NormalVGG((self.img_g + 1) * 0.5)
        self.loss_G_lp += (self.LPIPSloss(norm_real, norm_fake).mean()) * self.opt.lambda_lp if self.opt.lambda_lp > 0 else 0

        self.loss_G = self.loss_G_GAN + self.loss_G_rec + self.loss_G_lp

        self.loss_G.backward()

    def optimize_parameters(self):
        """update network weights"""
        # forward
        self.forward()
        # update D
        self.set_requires_grad([self.netD], True)
        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()
        # update G
        self.set_requires_grad([self.netD], False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

    def _refine_opt(self, opt):
        """modify the opt for refine generator and discriminator"""
        opt.netG = 'refine'
        opt.netD = 'style'
        opt.attn_D = True

        return opt
