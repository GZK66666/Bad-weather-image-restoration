import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import weights_init


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
        self.loss_names = ['G_clean', 'D_clean', 'idt_clean', 'kl_clean', 'G_badweather', 'D_badweather', 'idt_badweather', 'kl_badweather', 'G_background', 'D_background']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_clean = ['clean', 'fake_badweather', 'rec_clean']
        visual_names_badweather = ['badweather', 'fake_clean', 'rec_badweather']
        if self.isTrain and self.opt.lambda_identity > 0.0:
                visual_names_clean.append('idt_clean')
                visual_names_badweather.append('idt_badweather')

        self.visual_names = visual_names_clean + visual_names_badweather  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['encBackground_clean', 'decBackground_clean', 'encBackground_badweather',
                                'encWeather_badweather', 'decBadweather_badweather', 'netD_clean', 'netD_badweather', 'netD_background']
        else:  # during test time, only load Gs
            self.model_names = ['encBackground_clean', 'decBackground_clean', 'encBackground_badweather',
                                'encWeather_badweather', 'decBadweather_badweather']

        # define networks (both Generators and discriminators)
        # clean domain：一个背景encoder和一个背景decoder
        # badweather domain：所有天气图像域共用一个背景encoder，不同的天气特征encoder和decoder
        self.encBackground_clean = networks.ContentEncoder(n_downsample=2, n_res=4, input_dim=opt.input_nc, dim=64, norm='bn', activ='relu', pad_type='reflect').cuda(self.gpu_ids[0])
        self.decBackground_clean = networks.Decoder(n_upsample=2, n_res=4, dim=self.encBackground_clean.output_dim, output_dim=opt.input_nc, res_norm='bn', activ='relu', pad_type='reflect').cuda(self.gpu_ids[0])
        self.encBackground_badweather = networks.ContentEncoder(n_downsample=2, n_res=4, input_dim=opt.input_nc, dim=64, norm='bn', activ='relu', pad_type='reflect').cuda(self.gpu_ids[0])
        self.encWeather_badweather = [networks.NoiseEncoder(n_downsample=2, input_dim=opt.input_nc, dim=64, style_dim=self.encBackground_badweather.output_dim, norm='bn', activ='relu', pad_type='reflect').cuda(self.gpu_ids[0]) for _ in range(self.badweather_domains)]
        self.decBadweather_badweather = [networks.Decoder(n_upsample=2, n_res=4, dim=2 * self.encBackground_badweather.output_dim, output_dim=opt.input_nc, res_norm='bn', activ='relu', pad_type='reflect').cuda(self.gpu_ids[0]) for _ in range(self.badweather_domains)]

        if self.isTrain:  # define discriminators
            # netD_clean：干净背景图像判别器 netD_badweather：恶劣天气图像生成器（list, 有多少个恶劣天气图像域就有多少个对应的生成器） netD_background：域对抗判别器
            self.netD_clean = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids[0]).cuda(self.gpu_ids[0])
            self.netD_badweather = [networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                                      opt.init_type, opt.init_gain, self.gpu_ids[0]).cuda(self.gpu_ids[0]) for _ in range(self.badweather_domains)]
            self.netD_background = networks.Dis_content().cuda(self.gpu_ids[0])

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_clean_pool = ImagePool(opt.pool_size)
            self.fake_badweather_pools = [ImagePool(opt.pool_size) for _ in range(self.badweather_domains)]

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # todo： MSELoss刻画两个tensor的距离，reduction=‘sum’意味着得到的是两个向量差的平方和，那么是求和好还是求平均好呢？
            self.criterionCls = torch.nn.MSELoss(reduction='sum')

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_encBackground_clean = torch.optim.Adam(self.encBackground_clean.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_encBackground_badweather = torch.optim.Adam(self.encBackground_badweather.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_encWeather_badweather = [torch.optim.Adam(self.encWeather_badweather[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(self.badweather_domains)]
            self.optimizer_decBackground_clean = torch.optim.Adam(self.decBackground_clean.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_decBadweather_badweather = [torch.optim.Adam(self.decBadweather_badweather[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(self.badweather_domains)]
            self.optimizer_D_clean = torch.optim.Adam(self.netD_clean.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_badweather = [torch.optim.Adam(self.netD_badweather[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for i in range(self.badweather_domains)]
            self.optimizer_D_background = torch.optim.Adam(self.netD_background.parameters(), lr=opt.lr / 2, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_encBackground_clean)
            self.optimizers.append(self.optimizer_encBackground_badweather)
            for optimizer in self.optimizer_encWeather_badweather:
                self.optimizers.append(optimizer)
            self.optimizers.append(self.optimizer_decBackground_clean)
            for optimizer in self.optimizer_decBadweather_badweather:
                self.optimizers.append(optimizer)
            self.optimizers.append(self.optimizer_D_clean)
            for optimizer in self.optimizer_D_badweather:
                self.optimizers.append(optimizer)
            self.optimizers.append(self.optimizer_D_background)

            # Network weight initialize
            self.encBackground_clean.apply(weights_init('kaiming'))
            self.encBackground_badweather.apply(weights_init('kaiming'))
            for enc in self.encWeather_badweather:
                enc.apply(weights_init('kaiming'))
            self.decBackground_clean.apply(weights_init('kaiming'))
            for dec in self.decBadweather_badweather:
                dec.apply(weights_init('kaiming'))
            self.netD_clean.apply(weights_init('gaussian'))
            for dis in self.netD_badweather:
                dis.apply(weights_init('gaussian'))
            self.netD_background.apply(weights_init('gaussian'))

            # initialize loss storage
            self.loss_G_clean, self.loss_G_badweather = 0, [0] * self.badweather_domains
            self.loss_D_clean, self.loss_D_badweather = 0, [0] * self.badweather_domains
            self.loss_cycle_clean, self.loss_cycle_badweather = 0, [0] * self.badweather_domains
            self.loss_idt_clean, self.loss_idt_badweather = 0, [0] * self.badweather_domains
            self.loss_kl_clean, self.loss_kl_badweather = 0, [0] * self.badweather_domains

    def set_input(self, input):
        input_A = input['A']
        self.clean.resize_(input_A.size()).copy_(input_A)
        self.domain_clean = input['DA'][0]
        self.label_clean = torch.zeros(1, self.badweather_domains + 1).scatter_(1, self.domain_clean.unsqueeze(-1).unsqueeze(-1), 1) \
            .cuda(self.gpu_ids[0])
        if self.isTrain:
            input_B = input['B']
            self.badweather.resize_(input_B.size()).copy_(input_B)
            self.domain_badweather = input['DB'][0] - 1
            self.label_badweather = torch.zeros(1, self.badweather_domains + 1).scatter_(1, (self.domain_badweather + 1).unsqueeze(-1).unsqueeze(-1), 1) \
                .cuda(self.gpu_ids[0])
        self.image_paths = input['path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # encode
        self.clean_background = self.encBackground_clean(self.clean)
        self.badweather_background = self.encBackground_badweather(self.badweather)
        self.badweather_weather = self.encWeather_badweather[self.domain_badweather](self.badweather)

        # decode (within domain)
        h_badweather_background_weather_cat = torch.cat((self.badweather_background, self.badweather_weather), 1)
        # noise_single = torch.randn(self.clean_background.size()).cuda(self.clean_background.data.get_device())
        # noise_cat = torch.randn(h_badweather_background_weather_cat.size()).cuda(h_badweather_background_weather_cat.data.get_device())
        self.idt_clean = self.decBackground_clean(self.clean_background)
        self.idt_badweather = self.decBadweather_badweather[self.domain_badweather](h_badweather_background_weather_cat)

        # decode (cross domain)
        h_clean_background_badweather_weather_cat = torch.cat((self.clean_background, self.badweather_weather), 1)
        self.fake_clean = self.decBackground_clean(self.badweather_background)
        self.fake_badweather = self.decBadweather_badweather[self.domain_badweather](h_clean_background_badweather_weather_cat)

        # encode again
        self.fake_clean_background = self.encBackground_clean(self.fake_clean)
        self.fake_badweather_background = self.encBackground_badweather(self.fake_badweather)
        self.fake_badweather_weather = self.encWeather_badweather[self.domain_badweather](self.fake_badweather)

        # reconstruction
        h_fake_clean_background_fake_badweather_weather_cat = torch.cat((self.fake_clean_background, self.fake_badweather_weather), 1)
        self.rec_clean = self.decBackground_clean(self.fake_badweather_background)
        self.rec_badweather = self.decBadweather_badweather[self.domain_badweather](h_fake_clean_background_fake_badweather_weather_cat)

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

    def backward_D(self):
        self.set_requires_grad([self.netD_clean, self.netD_badweather[self.domain_badweather], self.netD_background], True)
        self.netD_clean.zero_grad()
        self.netD_badweather[self.domain_badweather].zero_grad()
        self.netD_background.zero_grad()

        # GAN loss
        fake_clean = self.fake_clean_pool.query(self.fake_clean)
        self.loss_D_clean = self.backward_D_basic(self.netD_clean, self.clean, fake_clean)
        fake_badweather = self.fake_badweather_pools[self.domain_badweather].query(self.fake_badweather)
        self.loss_D_badweather[self.domain_badweather] = self.backward_D_basic(self.netD_badweather[self.domain_badweather], self.badweather, fake_badweather)

        # domain adversarial loss
        pred_cls_clean = self.netD_background(self.clean_background)
        pred_cls_badweather = self.netD_background(self.badweather_background)
        self.loss_D_background = self.criterionCls(pred_cls_clean, self.label_clean) + self.criterionCls(pred_cls_badweather, self.label_badweather)
        self.loss_D_background.backward(retain_graph=True)
        # 修剪梯度 LIR代码中有的，可以考虑是否保留
        torch.nn.utils.clip_grad_norm_(self.netD_background.parameters(), 5)

        # optimizer step
        self.optimizer_D_clean.step()
        self.optimizer_D_badweather[self.domain_badweather].step()
        self.optimizer_D_background.step()

    def backward_D_Background(self):
        self.set_requires_grad(self.netD_background, True)
        self.netD_background.zero_grad()

        pred_cls_clean = self.netD_background(self.clean_background)
        pred_cls_badweather = self.netD_background(self.badweather_background)
        self.loss_D_background = self.criterionCls(pred_cls_clean, self.label_clean) + self.criterionCls(pred_cls_badweather, self.label_badweather)
        self.loss_D_background.backward(retain_graph=True)
        # 修剪梯度 LIR代码中有的，可以考虑是否保留
        torch.nn.utils.clip_grad_norm_(self.netD_background.parameters(), 5)

        self.optimizer_D_background.step()


    def backward_G(self):
        # zero grad
        self.set_requires_grad([self.netD_clean, self.netD_badweather[self.domain_badweather], self.netD_background], False)
        self.encBackground_clean.zero_grad()
        self.encBackground_badweather.zero_grad()
        self.encWeather_badweather[self.domain_badweather].zero_grad()
        self.decBackground_clean.zero_grad()
        self.decBadweather_badweather[self.domain_badweather].zero_grad()

        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_kl = self.opt.lambda_kl

        # Identity loss
        if lambda_idt > 0:
            self.loss_idt_clean = self.criterionIdt(self.idt_clean, self.clean) * lambda_B * lambda_idt
            self.loss_idt_badweather[self.domain_badweather] = self.criterionIdt(self.idt_badweather, self.badweather) * lambda_A * lambda_idt
        else:
            self.loss_idt_clean = 0
            self.loss_idt_badweather[self.domain_badweather] = 0

        # GAN loss
        self.loss_G_clean = self.criterionGAN(self.netD_clean(self.fake_clean), True)
        self.loss_G_badweather[self.domain_badweather] = self.criterionGAN(self.netD_badweather[self.domain_badweather](self.fake_badweather), True)

        # cycle loss
        self.loss_cycle_clean = self.criterionCycle(self.rec_clean, self.clean) * lambda_A
        self.loss_cycle_badweather[self.domain_badweather] = self.criterionCycle(self.rec_badweather, self.badweather) * lambda_B

        # domain adversarial loss
        pred_cls_clean = self.netD_background(self.clean_background)
        pred_cls_badweather = self.netD_background(self.badweather_background)
        # todo：target label设为[0.25, 0.25, 0.25, 0.25]，这样与[1, 0, 0, 0] ... [0, 0, 0, 1]的欧式距离都相等
        target_label = torch.Tensor([[0.25, 0.25, 0.25, 0.25]]).cuda(self.gpu_ids[0])
        self.loss_G_background = self.criterionCls(pred_cls_clean, target_label) + self.criterionCls(pred_cls_badweather, target_label)

        # kl loss
        loss_kl_clean_background = self.__compute_kl(self.clean_background)
        loss_kl_fake_clean_background = self.__compute_kl(self.fake_clean_background)
        self.loss_kl_clean = (loss_kl_clean_background + loss_kl_fake_clean_background) * lambda_kl
        loss_kl_badweather_background = self.__compute_kl(self.badweather_background)
        loss_kl_badweather_weather = self.__compute_kl(self.badweather_weather)
        loss_kl_fake_badweather_background = self.__compute_kl(self.fake_badweather_background)
        loss_kl_fake_badweather_weather = self.__compute_kl(self.fake_badweather_weather)
        self.loss_kl_badweather[self.domain_badweather] = (loss_kl_badweather_background + loss_kl_badweather_weather +
                                                           loss_kl_fake_badweather_background + loss_kl_fake_badweather_weather) * lambda_kl
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_clean + self.loss_G_badweather[self.domain_badweather] + self.loss_cycle_clean + \
                      self.loss_cycle_badweather[self.domain_badweather] + self.loss_idt_clean + \
                      self.loss_idt_badweather[self.domain_badweather] + self.loss_G_background + \
                      self.loss_kl_clean + self.loss_kl_badweather[self.domain_badweather]
        self.loss_G.backward()

        # optimizers step
        self.optimizer_encBackground_clean.step()
        self.optimizer_encBackground_badweather.step()
        self.optimizer_encWeather_badweather[self.domain_badweather].step()
        self.optimizer_decBackground_clean.step()
        self.optimizer_decBadweather_badweather[self.domain_badweather].step()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        # for _ in range(3):
        #     self.backward_D_Background()
        # todo：探究D_background是否可以将所有badweather domain当成一个域？
        self.backward_D()
        self.backward_G()

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def test(self):
        with torch.no_grad():
            # 1. clean input
            if self.domain_clean == 0:
                self.visuals = [self.clean]
                self.labels = ['clean']
                # encode
                clean_background = self.encBackground_clean(self.clean)
                # decode
                idt_clean = self.decBackground_clean(clean_background)
                self.visuals.append(idt_clean)
                self.labels.append('idt_clean')
            # 2. bad weather input
            else:
                self.visuals = [self.clean]
                self.labels = ['bad_weather']
                # encode
                badweather_background = self.encBackground_badweather(self.clean)
                badweather_weather = self.encWeather_badweather[self.domain_clean - 1](self.clean)
                # decode (within domain)
                h_badweather_background_weather_cat = torch.cat((badweather_background, badweather_weather), 1)
                idt_badweather = self.decBadweather_badweather[self.domain_clean - 1](h_badweather_background_weather_cat)
                self.visuals.append(idt_badweather)
                self.labels.append('idt_badweather')
                # decode (cross domain)
                fake_clean = self.decBackground_clean(badweather_background)
                self.visuals.append(fake_clean)
                self.labels.append('fake_clean')
                # encode again
                fake_clean_background = self.encBackground_clean(fake_clean)
                # decode again
                h_fake_clean_background_badweather_weather_cat = torch.cat((fake_clean_background, badweather_weather), 1)
                rec_badweather = self.decBadweather_badweather[self.domain_clean - 1](h_fake_clean_background_badweather_weather_cat)
                self.visuals.append(rec_badweather)
                self.labels.append('rec_badweather')

# todo：1，D和G的backward 2，模型的保存和读取以及测试过程