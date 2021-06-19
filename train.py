import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Faster_rcnn_frame import FasterRCNN
from trainer import FasterRCNNTrainer
from VOC_Dataset import VOC2012DataSet, frcnn_dataset_collate
from utils import LossHistory, weights_init
#from dataloader import FRCNNDataset, frcnn_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_ont_epoch(net,epoch,epoch_size_train,epoch_size_val,train_dataloader,val_dataloader,Epoch,cuda):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0
    with tqdm(total=epoch_size_train,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):
            if iteration >= epoch_size_train:
                break
            imgs, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor).cuda()
                else:
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            # print(labels)
            # print(boxes)
            losses = train_util.train_step(imgs, boxes, labels, 1)
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
            total_loss += total.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()
            
            pbar.set_postfix(**{'total'    : total_loss / (iteration + 1), 
                                'rpn_loc'  : rpn_loc_loss / (iteration + 1),  
                                'rpn_cls'  : rpn_cls_loss / (iteration + 1), 
                                'roi_loc'  : roi_loc_loss / (iteration + 1), 
                                'roi_cls'  : roi_cls_loss / (iteration + 1), 
                                'lr'       : get_lr(optimizer)})
            pbar.update(1)
            break#########
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataloader):
            if iteration >= epoch_size_val:
                break
            imgs,boxes,labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor).cuda()
                else:
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor)

                train_util.optimizer.zero_grad()
                losses = train_util.forward(imgs, boxes, labels, 1)
                _, _, _, _, val_total = losses

                val_toal_loss += val_total.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1)})
            pbar.update(1)

    #loss_history.append_loss(total_loss/(epoch_size_train+1), val_toal_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size_train+1),val_toal_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model,args.train_weight_dir+'train_model.pth')
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size_train+1),val_toal_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='..', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--voc_weights_vgg16', default='./model_weight/voc_weights_vgg.pth', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--train_weight_dir', default='./model_weight/', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的冻结的epoch数
    parser.add_argument('--f_epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的解冻的epoch数
    parser.add_argument('--uf_epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的冻结的batch size
    parser.add_argument('--f_batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    # 训练的解冻的batch size
    parser.add_argument('--uf_batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    # 设置冻结的学习率
    parser.add_argument('--f_lr', default=1e-4, type=float, help='learning rate')
    # 设置冻结的学习率
    parser.add_argument('--uf_lr', default=1e-4, type=float, help='learning rate')


    args = parser.parse_args()

    #----------------------创建模型
    model = FasterRCNN(args.num_classes)
    weights_init(model)

    #----------------------加载权重
    # model_weights_path = args.voc_weights_vgg16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # state_dict = torch.load(model_weights_path, map_location=device)
    # model.load_state_dict(state_dict)
    model.to(device)
    net = model.train()
    net = net.cuda()

    #---------------------训练准备
    optimizer       = optim.Adam(net.parameters(), args.f_lr, weight_decay=5e-4)
    lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

    train_dataset   = VOC2012DataSet(voc_root=args.data_path)
    val_dataset     = VOC2012DataSet(voc_root=args.data_path, txt_name='val.txt')
    train_dataloader= DataLoader(train_dataset, shuffle=False, batch_size=args.f_batch_size, num_workers=0, pin_memory=True,
                            drop_last=True, collate_fn=frcnn_dataset_collate)
    val_dataloader  = DataLoader(val_dataset, shuffle=True, batch_size=args.f_batch_size, num_workers=0, pin_memory=True,
                            drop_last=True, collate_fn=frcnn_dataset_collate)
                    
    epoch_size_train= len(train_dataset) // args.f_batch_size
    epoch_size_val  = len(val_dataset) // args.f_batch_size

    if epoch_size_train == 0 or epoch_size_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    for param in model.extractor.parameters():
        param.requires_grad = False

    # ------------------------------------#
    #   冻结bn层
    # ------------------------------------#
    model.freeze_bn()



    train_util = FasterRCNNTrainer(model, optimizer)
    for epoch in range(args.start_epoch,args.f_epochs):
        fit_ont_epoch(net,epoch,epoch_size_train,epoch_size_val,train_dataloader,val_dataloader,args.f_epochs,device==torch.device('cuda'))
        lr_scheduler.step()


    
    epoch_size_train= len(train_dataset) // args.uf_batch_size
    epoch_size_val  = len(val_dataset) // args.uf_batch_size
    optimizer       = optim.Adam(net.parameters(), args.uf_lr, weight_decay=5e-4)
    lr_scheduler    = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    #------------------------------------#
    #   解冻后训练
    #------------------------------------#
    for param in model.extractor.parameters():
        param.requires_grad = True

    # ------------------------------------#
    #   冻结bn层
    # ------------------------------------#
    model.freeze_bn()
    train_util = FasterRCNNTrainer(model,optimizer)

    for epoch in range(args.f_epochs, args.uf_epochs):
        fit_ont_epoch(net,epoch,epoch_size_val,epoch_size_val,train_dataloader,val_dataloader,args.uf_epochs, device==torch.device('cuda'))
        lr_scheduler.step()