from data_provider.data_factory_new import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visualize_losses
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# 以后记得每一次都新建一个Tensorboard文件夹来储存过程内容
# writer = SummaryWriter(log_dir='./runs/726_see_data_pipeline')

class Exp_new(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer_train(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        # 在这里实例化train vali test
        train_data, train_loader = self._get_data(flag='train')
        # val_data, val_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        # print(train_data.train_keys)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer_train()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        total_train_loss = []
        total_test_loss = []
        total_val_loss = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()

            # 确保模型处于训练模式
            self.model.train()

            for i, data in enumerate(train_loader):
                batch_x_qd = data['qd']
                batch_x_ir = data['ir']
                batch_x_qv = data['qv']
                batch_x_tmean = data['tmean']
                batch_y = data['target']

                iter_count += 1
                model_optim.zero_grad()

                outputs = self.model(batch_x_qd.to(self.device), batch_x_ir.to(self.device),
                                     batch_x_qv.to(self.device), batch_x_tmean.to(self.device)).to(self.device)
                batch_y = batch_y.to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 5000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            trainloss = np.average(train_loss)
            total_train_loss.append(float(trainloss))

            early_stopping(trainloss, self.model, path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            testloss = self.test(setting, test=1)
            total_test_loss.append(testloss.item())
            valloss = self.val(setting, val=1)
            total_val_loss.append(valloss.item())

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f} Val Loss: {4:.7f}".format(
                epoch + 1, train_steps, trainloss, testloss, valloss))
            # 写入tensorboard
            # for name, weight in self.model.named_parameters():
            #     writer.add_histogram(name, weight, epoch)
            #     writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path, 'losses_plot.pdf')

            visualize_losses(total_train_loss, total_test_loss, total_val_loss, save_path)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.to(self.device)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = self._select_criterion()
        total_test_loss = []
        # predict = []
        # ground_truth = []

        with torch.no_grad():
            for key in test_data.test_keys:
                test_loss = []
                predict = []
                ground_truth = []

                target = torch.tensor((test_data.selected_target[key]), dtype=torch.float)
                qd = torch.tensor(list(test_data.selected_qd[key].values()), dtype=torch.float)
                ir = torch.tensor(list(test_data.selected_ir[key].values()), dtype=torch.float)
                qv = torch.tensor(list(test_data.selected_qv[key].values()), dtype=torch.float)
                tmean = torch.tensor(list(test_data.selected_tmean[key].values()), dtype=torch.float)

                target = target.to(self.device)
                qd = qd.to(self.device)
                ir = ir.to(self.device)
                qv = qv.to(self.device)
                tmean = tmean.to(self.device)

                outputs = self.model(qd, ir, qv, tmean)
                loss = criterion(outputs, target)
                test_loss.append(loss.item())
                if target.dim() == outputs.dim() - 1:
                    target = target.unsqueeze(-1)

                predict.append(outputs.cpu().numpy())
                ground_truth.append(target.cpu().numpy())

                testloss = np.average(test_loss)
                predict = np.array(predict)
                ground_truth = np.array(ground_truth)

                # 将图片和表格保存到./test_results
                folder_path = './test_results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                visual(ground_truth.reshape(-1), predict.reshape(-1),
                       os.path.join(folder_path, 'Capacity' + key + '.pdf'))

                # 将数组保存到./results
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # 保存预测结果
                df = pandas.DataFrame({'all_pd': predict.reshape(-1)})
                # 定义要保存到的文件路径
                csv_filename = './test_results/predict' + key + '.csv'
                # 保存 DataFrame 到 CSV 文件中
                df.to_csv(csv_filename, index=False)

                mae, mse, rmse, mape, mspe = metric(predict, ground_truth)
                print('mse:{}, mae:{}, rmse:{}, battery:{}'.format(mse, mae, rmse, key))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}, rmse:{}, battery:{}'.format(mse, mae, rmse, key))
                f.write('\n')
                f.write('\n')
                f.close()

                np.save(folder_path + key + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path + key + 'pred.npy', predict)
                np.save(folder_path + key + 'true.npy', ground_truth)

        return testloss

        #     for i, data in enumerate(test_loader):
        #         batch_x_qd = data['qd'].to(self.device)
        #         batch_x_ir = data['ir'].to(self.device)
        #         batch_x_qv = data['qv'].to(self.device)
        #         batch_x_tmean = data['tmean'].to(self.device)
        #         batch_y = data['target'].to(self.device)
        #
        #         outputs = self.model(batch_x_qd, batch_x_ir, batch_x_qv, batch_x_tmean)
        #         loss = criterion(outputs, batch_y)
        #         test_loss.append(loss.item())
        #         if batch_y.dim() == outputs.dim() - 1:
        #             batch_y = batch_y.unsqueeze(-1)
        #
        #         predict.append(outputs.cpu().numpy())
        #         ground_truth.append(batch_y.cpu().numpy())
        #
        # testloss = np.average(test_loss)
        # predict = np.array(predict)
        # ground_truth = np.array(ground_truth)
        # visual(ground_truth.reshape(-1), predict.reshape(-1), os.path.join(folder_path, 'Capacity' + '.pdf'))
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # 保存预测结果
        df = pandas.DataFrame({'all_pd': predict.reshape(-1)})
        # 定义要保存到的文件路径
        csv_filename = './test_results/predict.csv'
        # 保存 DataFrame 到 CSV 文件中
        df.to_csv(csv_filename, index=False)

        mae, mse, rmse, mape, mspe = metric(predict, ground_truth)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', predict)
        np.save(folder_path + 'true.npy', ground_truth)

        return testloss

    def val(self, setting, val=0):
        val_data, val_loader = self._get_data(flag='val')

        # print(val_data.val_keys)

        if val:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.to(self.device)

        folder_path = './val_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = self._select_criterion()

        val_loss = []

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                batch_x_qd = data['qd'].to(self.device)
                batch_x_ir = data['ir'].to(self.device)
                batch_x_qv = data['qv'].to(self.device)
                batch_x_tmean = data['tmean'].to(self.device)
                batch_y = data['target'].to(self.device)

                outputs = self.model(batch_x_qd, batch_x_ir, batch_x_qv, batch_x_tmean)
                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())
                if batch_y.dim() == outputs.dim() - 1:
                    batch_y = batch_y.unsqueeze(-1)

        valloss = np.average(val_loss)

        return valloss