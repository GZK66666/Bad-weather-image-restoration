import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from util import util


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.pre_training_dir = os.path.join(opt.pre_training_model_dir, opt.pre_training_model_name)
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.visuals = []
        self.labels = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.badweather_domains = opt.badweather_domains
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            self.load_preTraining_networks()
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        if self.isTrain:
            for name in self.visual_names:
                if isinstance(name, str):
                    visual_ret[name] = getattr(self, name)
        else:
            images =[util.tensor2im(v.data) for v in self.visuals]
            visual_ret =OrderedDict(zip(self.labels, images))
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        for name in self.loss_names:
            cur_loss = getattr(self, 'loss_' + name)
            if isinstance(cur_loss, list):
                errors_ret[name] = extract(cur_loss)
            else:
                errors_ret[name] = float(cur_loss)

        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, list):
                        for i in range(len(net)):
                            save_filename = '%s_net_%s' % (epoch, name)
                            save_filename += str(i) + '.pth'
                            save_path = os.path.join(self.save_dir, save_filename)
                            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                                torch.save(net[i].module.cpu().state_dict(), save_path)
                                net[i].cuda(self.gpu_ids[0])
                            else:
                                torch.save(net[i].cpu().state_dict(), save_path)
                else:
                    save_filename = '%s_net_%s' % (epoch, name)
                    save_filename += '.pth'
                    save_path = os.path.join(self.save_dir, save_filename)
                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(net.module.cpu().state_dict(), save_path)
                        net.cuda(self.gpu_ids[0])
                    else:
                        torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if name == 'G_clean':
                    # 加载G_clean
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    load_path = os.path.join(self.save_dir, load_filename)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)
                else :
                    # 加载G_badweather
                    weather = ['smallrain', 'raindrop', 'snow']
                    for i in range(3):
                        load_filename = 'pre_training_G_%s.pth' % weather[i]
                        load_path = os.path.join(self.pre_training_dir, load_filename)
                        if isinstance(net[i], torch.nn.DataParallel):
                            net[i] = net[i].module
                        print('loading the preTraining model from %s' % load_path)
                        state_dict = torch.load(load_path, map_location=str(self.device))
                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata
                        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                            self.__patch_instance_norm_state_dict(state_dict, net[i], key.split('.'))
                        net[i].load_state_dict(state_dict)

    def load_preTraining_networks(self):
        weather = ['smallrain', 'raindrop', 'snow']
        # 加载预训练的恶劣天气生成器和判别器
        netGs = getattr(self, 'netG_badweather')
        netDs = getattr(self, 'netD_badweather')
        for i in range(3):
            load_G_filename = 'pre_training_G_%s.pth' % weather[i]
            load_G_path = os.path.join(self.pre_training_dir, load_G_filename)
            load_D_filename = 'pre_training_D_%s.pth' % weather[i]
            load_D_path = os.path.join(self.pre_training_dir, load_D_filename)
            if isinstance(netGs[i], torch.nn.DataParallel):
                netGs[i] = netGs[i].module
            if isinstance(netDs[i], torch.nn.DataParallel):
                netDs[i] = netDs[i].module
            print('loading the preTraining generator from %s' % load_G_path)
            print('loading the preTraining discriminator from %s' % load_D_path)
            state_dict_G = torch.load(load_G_path, map_location=str(self.device))
            state_dict_D = torch.load(load_D_path, map_location=str(self.device))
            if hasattr(state_dict_G, '_metadata'):
                del state_dict_G._metadata
            if hasattr(state_dict_D, '_metadata'):
                del state_dict_D._metadata
            for key in list(state_dict_G.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict_G, netGs[i], key.split('.'))
            for key in list(state_dict_D.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict_D, netDs[i], key.split('.'))
            netGs[i].load_state_dict(state_dict_G)
            netDs[i].load_state_dict(state_dict_D)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                if isinstance(net, list):
                    for i in range(len(net)):
                        num_params = 0
                        for param in net[i].parameters():
                            num_params += param.numel()
                        if (verbose):
                            print(net[i])
                        print('[Network %s] Total number of parameters : %.3f M' % (name + str(i), num_params / 1e6))
                else:
                    for param in net.parameters():
                        num_params += param.numel()
                    if verbose:
                        print(net)
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
