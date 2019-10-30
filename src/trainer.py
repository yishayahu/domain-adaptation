import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models as models
import utils
import abc
from train_result import FitResult, EpochResult, BatchResult
import os
from pathlib import Path
from typing import Callable, Any
import sys
import tqdm
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, opt, num_classes, source_train_ds, source_test_ds):
        self.source_train_ds = source_train_ds
        self.source_test_ds = source_test_ds
        self.opt = opt
        self.best_val = 0
        self.num_classes = num_classes

        # networks and optimizers
        self.mixer = models.Mixer(opt)
        self.classifier = models.Classifier(opt, num_classes)

        # initialize weight's
        self.mixer.apply(utils.weights_init)
        self.classifier.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion = nn.CrossEntropyLoss()

        if opt.gpu >= 0:
            self.mixer.cuda()
            self.classifier.cuda()
            self.criterion.cuda()

        # Defining optimizers
        self.optimizer_mixer = optim.Adam(self.mixer.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def fit(self, dl_train, dl_test,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement = saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            # - Use the train/test_epoch methods.
            # - Save losses and accuracies in the lists above.
            # - Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            batches = None
            if "max_batches" in kw:
                batches = kw.get("max_batches")
            train_epoch_res = self.train_epoch(dl_train,epoch=epoch, verbose=verbose, max_batches=batches)
            test_epoch_res = self.test_epoch(dl_test, verbose=verbose, max_batches=batches)

            train_loss.append(torch.mean(torch.Tensor(train_epoch_res.losses)).item())
            train_acc.append(train_epoch_res.accuracy)
            test_loss.append(torch.mean(torch.Tensor(test_epoch_res.losses)).item())
            test_acc.append(test_epoch_res.accuracy)
            if checkpoints and (len(test_acc) == 1 or test_acc[-2] < test_acc[-1]):
                torch.save(self.model, checkpoints)
            if len(test_acc) == 1 or test_acc[-2] < test_acc[-1]:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping and epochs_without_improvement >= early_stopping:
                    actual_num_epochs = epoch + 1
                    break
            actual_num_epochs = epoch + 1
            train_result = EpochResult(losses=train_loss[-1], accuracy=train_acc[-1])
            test_result = EpochResult(losses=test_loss[-1], accuracy=test_acc[-1])
            if len(test_acc) == 1 or (len(test_acc) >= 2 and test_acc[-1] > test_acc[-2]):
                save_checkpoint = True
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc, ewi=epochs_without_improvement, model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch + 1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train,epoch, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        raise NotImplemented("Trainer is an abstract class")
        # todo: use _foreecah bach wile implemnting

    def test_epoch(self, dl_test, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        raise NotImplemented("Trainer is an abstract class")

    @abc.abstractmethod
    def train_batch(self, batch, epoch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplemented("Trainer is an abstract class")

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplemented("Trainer is an abstract class")

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl, forward_fn: Callable[[Any, Any], BatchResult], epoch, verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data, epoch)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class SourceTrainer(Trainer):
    def __init__(self, opt, num_classes, source_train_ds, source_test_ds):
        super().__init__(opt, num_classes, source_train_ds, source_test_ds)

    def train_epoch(self, dl_train: DataLoader, epoch, **kw) -> EpochResult:
        self.mixer.train()
        self.classifier.train()
        res = self._foreach_batch(dl_train, self.train_batch, epoch=epoch)
        # Validate every epoch  # todo: i erased validate because i think my func do the same
        return res

    def train_batch(self, batch, epoch) -> BatchResult:
        ###########################
        # Forming input variables
        ###########################

        src_inputs, src_labels = batch
        if self.opt.gpu >= 0:
            src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
        src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)

        ###########################
        # Updates
        ###########################

        self.classifier.zero_grad()
        self.mixer.zero_grad()
        outC = self.classifier(self.mixer(src_inputsv))
        loss = self.criterion(outC, src_labelsv)
        num_of_correct = 0  # todo: change num of correct
        loss.backward()
        self.optimizer_classifier.step()
        self.optimizer_mixer.step()

        # Learning rate scheduling
        if self.opt.lrd:
            print(epoch)
            # todo: i changed curr_iter to epoch
            self.optimizer_mixer = utils.exp_lr_scheduler(self.optimizer_mixer, self.opt.lr, self.opt.lrd, epoch)
            self.optimizer_classifier = utils.exp_lr_scheduler(self.optimizer_classifier, self.opt.lr, self.opt.lrd, epoch)
        return BatchResult(loss.item(), int(num_of_correct/len(outC)))


class FullTrainer(Trainer):
    def __init__(self, opt, num_classes, source_train_ds, source_test_ds, target_train_ds, mean, std):
        super().__init__(opt, num_classes, source_train_ds, source_test_ds)

        self.target_train_ds = target_train_ds
        self.mean = mean
        self.std = std

        # Defining networks and optimizers

        self.generator = models.Generator(opt, num_classes)
        self.discriminator = models.Discriminator(opt, num_classes)

        # Weight initialization
        self.generator.apply(utils.weights_init)
        self.discriminator.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if opt.gpu >= 0:
            self.discriminator.cuda()
            self.generator.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

    """
    Validation function
    """
    def validate(self, epoch):

        self.mixer.eval()
        self.classifier.eval()
        total = 0
        correct = 0

        # Testing the model
        for i, datas in enumerate(self.source_test_ds):
            inputs, labels = datas
            inputv, labelv = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda())

            outC = self.classifier(self.mixer(inputv))
            _, predicted = torch.max(outC.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        val_acc = 100 * float(correct) / total
        print('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))

        # Saving checkpoints
        torch.save(self.mixer.state_dict(), '%s/models/netF.pth' % (self.opt.outf))
        torch.save(self.classifier.state_dict(), '%s/models/netC.pth' % (self.opt.outf))

        if val_acc > self.best_val:
            self.best_val = val_acc
            torch.save(self.mixer.state_dict(), '%s/models/model_best_netF.pth' % (self.opt.outf))
            torch.save(self.classifier.state_dict(), '%s/models/model_best_netC.pth' % (self.opt.outf))

    """
    Train function
    """

    def train(self):

        curr_iter = 0

        reallabel = torch.FloatTensor(self.opt.batchSize).fill_(self.real_label_val)
        fakelabel = torch.FloatTensor(self.opt.batchSize).fill_(self.fake_label_val)
        if self.opt.gpu >= 0:
            reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()
        reallabelv = Variable(reallabel)
        fakelabelv = Variable(fakelabel)

        for epoch in range(self.opt.nepochs):
            print("epoch = ", epoch)
            self.generator.train()
            self.mixer.train()
            self.classifier.train()
            self.discriminator.train()

            for i, (datas, datat) in enumerate(itertools.zip_longest(self.source_train_ds, self.target_train_ds)):
                if i >= min(len(self.source_train_ds), len(self.target_train_ds)):
                    # todo: check what to do with the left over data
                    break

                ###########################
                # Forming input variables
                ###########################

                src_inputs, src_labels = datas
                tgt_inputs, __ = datat
                src_inputs_unnorm = (((src_inputs * self.std[0]) + self.mean[0]) - 0.5) * 2

                # Creating one hot vector
                labels_onehot = np.zeros((self.opt.batchSize, self.num_classes + 1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((self.opt.batchSize, self.num_classes + 1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, self.num_classes] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)

                if self.opt.gpu >= 0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                    src_inputs_unnorm = src_inputs_unnorm.cuda()
                    tgt_inputs = tgt_inputs.cuda()
                    src_labels_onehot = src_labels_onehot.cuda()
                    tgt_labels_onehot = tgt_labels_onehot.cuda()

                # Wrapping in variable
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                src_inputs_unnormv = Variable(src_inputs_unnorm)
                tgt_inputsv = Variable(tgt_inputs)
                src_labels_onehotv = Variable(src_labels_onehot)
                tgt_labels_onehotv = Variable(tgt_labels_onehot)

                ###########################
                # Updates
                ###########################

                # Updating D network

                self.discriminator.zero_grad()
                src_emb = self.mixer(src_inputsv)
                src_emb_cat = torch.cat((src_labels_onehotv, src_emb), 1)
                src_gen = self.generator(src_emb_cat)

                tgt_emb = self.mixer(tgt_inputsv)
                tgt_emb_cat = torch.cat((tgt_labels_onehotv, tgt_emb), 1)
                tgt_gen = self.generator(tgt_emb_cat)

                src_realoutputD_s, src_realoutputD_c = self.discriminator(src_inputs_unnormv)
                errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv)
                errD_src_real_c = self.criterion_c(src_realoutputD_c, src_labelsv)

                src_fakeoutputD_s, src_fakeoutputD_c = self.discriminator(src_gen)
                errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.discriminator(tgt_gen)
                errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s
                errD.backward(retain_graph=True)
                self.optimizer_discriminator.step()

                # Updating G network

                self.generator.zero_grad()
                src_fakeoutputD_s, src_fakeoutputD_c = self.discriminator(src_gen)
                errG_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                errG_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                errG = errG_c + errG_s
                errG.backward(retain_graph=True)
                self.optimizer_generator.step()

                # Updating C network

                self.classifier.zero_grad()
                outC = self.classifier(src_emb)
                errC = self.criterion_c(outC, src_labelsv)
                errC.backward(retain_graph=True)
                self.optimizer_classifier.step()

                # Updating F network

                self.mixer.zero_grad()
                errF_fromC = self.criterion_c(outC, src_labelsv)

                src_fakeoutputD_s, src_fakeoutputD_c = self.discriminator(src_gen)
                errF_src_fromD = self.criterion_c(src_fakeoutputD_c, src_labelsv) * (self.opt.adv_weight)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.discriminator(tgt_gen)
                errF_tgt_fromD = self.criterion_s(tgt_fakeoutputD_s, reallabelv) * (
                            self.opt.adv_weight * self.opt.alpha)

                errF = errF_fromC + errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizer_mixer.step()

                curr_iter += 1

                # Visualization
                if i == 1:
                    vutils.save_image((src_gen.data / 2) + 0.5,
                                      '%s/visualization/source_gen_%d.png' % (self.opt.outf, epoch))
                    vutils.save_image((tgt_gen.data / 2) + 0.5,
                                      '%s/visualization/target_gen_%d.png' % (self.opt.outf, epoch))

                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizer_discriminator = utils.exp_lr_scheduler(self.optimizer_discriminator, self.opt.lr, self.opt.lrd,
                                                             curr_iter)
                    self.optimizer_mixer = utils.exp_lr_scheduler(self.optimizer_mixer, self.opt.lr, self.opt.lrd,
                                                             curr_iter)
                    self.optimizer_classifier = utils.exp_lr_scheduler(self.optimizer_classifier, self.opt.lr, self.opt.lrd,
                                                             curr_iter)

                    # Validate every epoch
            self.validate(epoch + 1)


