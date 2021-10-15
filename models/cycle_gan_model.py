import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.clean = self.Tensor(opt.batch_size, opt.input_nc, opt.crop_size, opt.crop_size)
        self.badweather = self.Tensor(opt.batch_size, opt.input_nc, opt.crop_size, opt.crop_size)
        # clean domain and bad weather domain
        self.domain_clean = None
        self.domain_badweather = None

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_clean', 'D_clean', 'idt_clean', 'G_badweather', 'D_badweather', 'idt_badweather']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_clean = ['clean', 'fake_badweather', 'rec_clean']
        visual_names_badweather = ['badweather', 'fake_clean', 'rec_badweather']
        if self.isTrain and self.opt.lambda_identity > 0.0:
                visual_names_clean.append('idt_clean')
                visual_names_badweather.append('idt_badweather')

        self.visual_names = visual_names_clean + visual_names_badweather  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_clean', 'G_badweather', 'D_clean', 'D_badweather']
        else:  # during test time, only load Gs
            self.model_names = ['G_clean', 'G_badweather']

        # define networks (both Generators and discriminators)
        # netG_clean：干净背景图像生成器 netG_badweather：恶劣天气图像生成器（list，有多少个恶劣天气图像域就有多少个对应的生成器）
        self.netG_clean = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_badweather = [networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not
        opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) for _ in range(self.badweather_domains)]

        if self.isTrain:  # define discriminators
            # netD_clean：干净背景图像判别器 netD_badweather：恶劣天气图像生成器（list, 有多少个恶劣天气图像域就有多少个对应的生成器）
            self.netD_clean = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_badweather = [networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                                      opt.init_type, opt.init_gain, self.gpu_ids) for _ in range(self.badweather_domains)]

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_clean_pool = ImagePool(opt.pool_size)
            self.fake_badweather_pools = [ImagePool(opt.pool_size) for _ in range(self.badweather_domains)]
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_clean = torch.optim.Adam(self.netG_clean.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_badwheather = [torch.optim.Adam(self.netG_badweather[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(self.badweather_domains)]
            self.optimizer_D_clean = torch.optim.Adam(self.netD_clean.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_badwheather = [torch.optim.Adam(self.netD_badweather[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(self.badweather_domains)]
            self.optimizers.append(self.optimizer_G_clean)
            for optimizer in self.optimizer_G_badwheather:
                self.optimizers.append(optimizer)
            self.optimizers.append(self.optimizer_D_clean)
            for optimizer in self.optimizer_D_badwheather:
                self.optimizers.append(optimizer)
            # initialize loss storage
            self.loss_G_clean, self.loss_G_badweather = 0, [0] * self.badweather_domains
            self.loss_D_clean, self.loss_D_badweather = 0, [0] * self.badweather_domains
            self.loss_cycle_clean, self.loss_cycle_badweather = 0, [0] * self.badweather_domains
            self.loss_idt_clean, self.loss_idt_badweather = 0, [0] * self.badweather_domains

    def set_input(self, input):
        input_A = input['A']
        self.clean.resize_(input_A.size()).copy_(input_A)
        self.domain_clean = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.badweather.resize_(input_B.size()).copy_(input_B)
            self.domain_badweather = input['DB'][0] - 1
        self.image_paths = input['path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_clean = self.netG_clean(self.badweather)
        self.fake_badweather = self.netG_badweather[self.domain_badweather](self.clean)
        self.rec_clean = self.netG_clean(self.fake_badweather)
        self.rec_badweather = self.netG_badweather[self.domain_badweather](self.fake_clean)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_Clean(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_clean = self.fake_clean_pool.query(self.fake_clean)
        self.loss_D_clean = self.backward_D_basic(self.netD_clean, self.clean, fake_clean)

    def backward_D_Badweather(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_badweather = self.fake_badweather_pools[self.domain_badweather].query(self.fake_badweather)
        self.loss_D_badweather[self.domain_badweather] = self.backward_D_basic(self.netD_badweather[self.domain_badweather], self.badweather, fake_badweather)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_clean should be identity if clean is fed: ||G_clean(clean) - clean||
            self.idt_clean = self.netG_clean(self.clean)
            self.loss_idt_clean = self.criterionIdt(self.idt_clean, self.clean) * lambda_B * lambda_idt
            # G_badweather should be identity if badweather is fed: ||G_badweather(badweather) - badweather||
            self.idt_badweather = self.netG_badweather[self.domain_badweather](self.badweather)
            self.loss_idt_badweather[self.domain_badweather] = self.criterionIdt(self.idt_badweather, self.badweather) * lambda_A * lambda_idt
        else:
            self.loss_idt_clean = 0
            self.loss_idt_badweather[self.domain_badweather] = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_clean = self.criterionGAN(self.netD_clean(self.fake_clean), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_badweather[self.domain_badweather] = self.criterionGAN(self.netD_badweather[self.domain_badweather](self.fake_badweather), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_clean = self.criterionCycle(self.rec_clean, self.clean) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_badweather[self.domain_badweather] = self.criterionCycle(self.rec_badweather, self.badweather) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_clean + self.loss_G_badweather[self.domain_badweather] + self.loss_cycle_clean + \
                      self.loss_cycle_badweather[self.domain_badweather] + self.loss_idt_clean + \
                      self.loss_idt_badweather[self.domain_badweather]
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
         # G_clean and G_badweather
        self.set_requires_grad([self.netD_clean, self.netD_badweather[self.domain_badweather]], False)
        self.netG_clean.zero_grad()
        self.netG_badweather[self.domain_badweather].zero_grad()
        self.backward_G()
        self.optimizer_G_clean.step()
        self.optimizer_G_badwheather[self.domain_badweather].step()
        # D_clean and D_badweather
        self.set_requires_grad([self.netD_clean, self.netD_badweather[self.domain_badweather]], True)
        self.netD_clean.zero_grad()
        self.netD_badweather[self.domain_badweather].zero_grad()
        self.backward_D_Clean()
        self.backward_D_Badweather()
        self.optimizer_D_clean.step()
        self.optimizer_D_badwheather[self.domain_badweather].step()

    def test(self):
        with torch.no_grad():
            # 1. clean input
            if self.domain_clean == 0:
                self.visuals = [self.clean]
                self.labels = ['clean']
                # idt
                idt_clean = self.netG_clean(self.clean)
                self.visuals.append(idt_clean)
                self.labels.append('idt_clean')
                # fake weather
                for i in range(self.badweather_domains):
                    badweather = self.netG_badweather[i](self.clean)
                    self.visuals.append(badweather)
                    self.labels.append('fake_bad_weather_%d' % i)
                    rec_clean = self.netG_clean(badweather)
                    self.visuals.append(rec_clean)
                    self.labels.append('rec_clean_%d' % i)
            # 2. bad weather input
            else:
                self.visuals = [self.clean]
                self.labels = ['bad_weather']
                # idt
                idt_bad_weather = self.netG_badweather[self.domain_clean - 1](self.clean)
                self.visuals.append(idt_bad_weather)
                self.labels.append('idt_bad_weather')
                # fake clean
                fake_clean = self.netG_clean(self.clean)
                self.visuals.append(fake_clean)
                self.labels.append('fake_clean')
                rec_bad_weather = self.netG_badweather[self.domain_clean - 1](fake_clean)
                self.visuals.append(rec_bad_weather)
                self.labels.append('rec_bad_weather')