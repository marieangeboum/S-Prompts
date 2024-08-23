import copy
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain
from models.sinet import SiNet
from models.slinet import SliNet

class SPrompts(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "slip":
            self._network = SliNet(args)
        elif args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))

        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.topk = 1   # origin is 5
        self.class_num = self._network.class_num
        self.all_keys = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task) # ??
        # donne l'indice de la session en cours pour récuperer le classifieur et les prompts associés
        self._network.update_fc(self._total_classes) # c'est quoi total_classes
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        # récupère les données d'entrainement : retourne le dataloader dummydataset
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')

        # création  du train loader Dataloader
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        val_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='val',
                                                 mode='val')
        # idem avec le jeu de test
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader) # Et là on appelle cette fonction pour lancer l'entraînement
        self.clustering(self.train_loader)  # clustering sur les features des données d'inputs

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            # self._network.module.numtask
            if isinstance(self._network, nn.DataParallel):
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                # if "prompt_pool" + "." + str(self._network.module.numtask - 1) in name:
                #     param.requires_grad_(True)
            else :
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                # if "prompt_pool" + "." + str(self._network.numtask - 1) in name:
                #     param.requires_grad_(True)
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        if self._cur_task==0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                logits = self._network(inputs)['masks']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy_domain(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain(y_pred.T[0], y_true, self._known_classes, class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret
 
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
                taskselection = []
                for task_centers in self.all_keys: # centroids des clusters Kmeans
                # task centers qui sont de taille 5, 768 5 pcq j'ai 5 kmeans par sessions
                    tmpcentersbatch = []
                    for center in task_centers: # je peux avoir plusieurs centroids par domaines
                    # pour chaque centre je calcule la distance entre les features et le centroide
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1)) # calcul de la distance euclidienne entre les features et les centroides
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0]) # je selecionne les centroides les plus proches de mes features 

                selection = torch.vstack(taskselection).min(0)[1] # je recupere les indices des centroides les plus proches pour chaque ^

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, selection)
                else:
                    outputs = self._network.interface(inputs, selection)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)